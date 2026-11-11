package d2g.floorplan.model

import d2g.*
import d2g.model.*
import d2s.model.*
import d2s.floorplan.model.D2S
import d2s.floorplan.model.D2S.NodeQueryLogits
import deepwit.attention.KVCache
import deepwit.base.AffineLayer
import deepwit.embedder.LearnedAbsolutePositionalInjector
import deepwit.embedder.VocabularyEmbedder
import deepwit.init.Init
import EdgeHead.EdgeLogits
import dimwit.*
import syntheticFloorplansDataset.FloorplanClass
import syntheticFloorplansDataset.FloorplanEdges
import syntheticFloorplansDataset.FloorplanRecord
import syntheticFloorplansDataset.FloorplanRelationship
import syntheticFloorplansDataset.FloorplanRelationships

import scala.language.implicitConversions

/** Document-to-graph model: a [[D2S]] for the nodes of a floor plan, and the relationships between
  * them on top.
  *
  * 1. The document (or: image) is embedded by a vision transformer to a sequence of patch embeddings.
  * 2. The graph is predicted in two stages:
  *   a. the nodes based on cross-attenting the document (1), by the [[D2S]]
  *   b. the relationships based on cross-attenting the document (1) and the nodes (2a).
  */
class D2G[V: IsFloating](params: D2G.Params[V]):

  import D2G.EdgeQueryLogits
  import D2G.Scores

  /** The model of the nodes, which the relationships are written on top of. */
  val set = D2S(params.set)
  val encodeDocument = set.encodeDocument
  val nodeHead = set.nodeHead

  private val embedEdges = EdgeEmbedder(params.edges.embedder)
  private val edgePosition = LearnedAbsolutePositionalInjector(params.edges.positions)
  private val edgeDecoder = EdgeDecoder(params.edges.decoder)
  val edgeHead = EdgeHead(params.edges.head)

  private val pool = set.pool

  /** What two queries of the pool answer. Queries selected randomly. */
  def logits(document: Tensor3[Width, Height, Channel, V], taken: FloorplanRecord[Node, Edge], asked: Key): Scores[V] =
    val randomQueryIds = Random.permutation(Axis[PoolQuery] -> pool)(asked).slice(Axis[PoolQuery].at(0 until 2))
    predict(encodeDocument(document), taken, randomQueryIds, randomQueryIds)

  /** What every query of the pool answers, in one reading. */
  def logitsPerQuery(encoded: Tensor2[Patch, Embedding, V], taken: FloorplanRecord[Node, Edge]): Scores[V] =
    val allQueryIds = Tensor1(Axis[PoolQuery], VType[Int32]).fromArray(Array.range(0, pool))
    predict(encoded, taken, allQueryIds, allQueryIds)

  /** What each asked query answers at every slot. */
  private def predict(
      encodedDocument: Tensor2[Patch, Embedding, V],
      taken: FloorplanRecord[Node, Edge],
      nodeQueryIds: Tensor1[PoolQuery, Int32],
      edgeQueryIds: Tensor1[PoolQuery, Int32]
  ): Scores[V] =

    val (carriedNodes, nodes) = set.predict(encodedDocument, taken.nodes, nodeQueryIds)

    val nodeSource = NodeSource(carriedNodes, !(taken.nodes.nodeClass elementEquals_! FloorplanClass.NoNode.id))
    val (_, answeredEdges) =
      val takenEdges = edgePosition(embedEdges(taken.edges, carriedNodes))
      val queryEdges = params.edges.queries.slice(Axis[PoolQuery].at(edgeQueryIds)) // take queries and broadcast along context
        .vmap(Axis[PoolQuery]): query =>
          edgePosition(query.broadcastTo(takenEdges.shape))
      edgeDecoder.forTraining(encodedDocument, nodeSource, takenEdges, queryEdges)

    val (relationship, subject, obj) =
      answeredEdges.vmap(Axis[PoolQuery]): answered =>
        val scored = edgeHead(answered, nodeSource)
        (scored.relationship, scored.subject, scored.obj)

    Scores(nodes, EdgeQueryLogits(relationship, subject, obj))

  /** Every edge decoder block's keys and values of the encoded document and of the written `nodes`,
    * which every relationship step reads.
    */
  def readForEdges(encodedDocument: Tensor2[Patch, Embedding, V], nodes: NodeSource[Embedding, V]): List[EdgeSources[V]] =
    edgeDecoder.read(encodedDocument, nodes)

  /** The edge decoder before any relationship is taken. */
  def noEdgeTaken(slots: AxisExtent[Edge]): List[KVCache[Edge, V]] =
    edgeDecoder.nothingTaken(slots)

  /** Takes `edge`, a record of one relationship between `nodes`, at `slot`, for the slots after it to read. */
  def takeEdge(nodes: NodeSource[Embedding, V], sources: List[EdgeSources[V]], taken: List[KVCache[Edge, V]], slot: Tensor0[Int32], edge: FloorplanEdges[Edge]): List[KVCache[Edge, V]] =
    edgeDecoder.take(sources, taken, slot, edgePosition.injectAt(slot)(embedEdges(edge, nodes.nodeEmbeddings).squeeze(Axis[Edge])))

  /** What the pool answers at relationship `slot`, given the relationships taken before it: per
    * query, in the order of the queries, the prediction embedding it ends as and the candidate
    * relationship between `nodes` the head reads from it.
    */
  def answerEdgeAt(nodes: NodeSource[Embedding, V], sources: List[EdgeSources[V]], taken: List[KVCache[Edge, V]], slot: Tensor0[Int32]): (Tensor2[PoolQuery, Embedding, V], EdgeLogits[V]) =
    val asked = params.edges.queries.vmap(Axis[PoolQuery])(edgePosition.injectAt(slot))
    val predictions = edgeDecoder.answer(sources, taken, slot, asked)
    (predictions, edgeHead(predictions.relabel(Axis[PoolQuery] -> Axis[Edge]), nodes))

object D2G:

  case class Scores[V](nodes: NodeQueryLogits[V], edges: EdgeQueryLogits[V])

  object Scores:

    given tensorTree[V]: TensorTree[Scores[V]] = TensorTree.derived
    given tree[V]: TreeOf[Scores[V], V] = TreeOf.derived

  /** [[EdgeLogits]] at every query slot. */
  case class EdgeQueryLogits[V](
      relationship: Tensor3[PoolQuery, Edge, FloorplanRelationships, V],
      subject: Tensor3[PoolQuery, Edge, Node, V],
      obj: Tensor3[PoolQuery, Edge, Node, V]
  ):
    def at(query: Int): EdgeLogits[V] = EdgeLogits(
      relationship.slice(Axis[PoolQuery].at(query)),
      subject.slice(Axis[PoolQuery].at(query)),
      obj.slice(Axis[PoolQuery].at(query))
    )

  object EdgeQueryLogits:

    given tensorTree[V]: TensorTree[EdgeQueryLogits[V]] = TensorTree.derived
    given tree[V]: TreeOf[EdgeQueryLogits[V], V] = TreeOf.derived

  case class Params[V](
      set: D2S.Params[V],
      edges: Params.EdgeParams[V]
  )

  object Params:

    given tensorTree: TensorTree[Params[Float32]] = TensorTree.derived
    given tree: TreeOf[Params[Float32], Float32] = TreeOf.derived

    case class EdgeParams[V](
        decoder: EdgeDecoder.Params[Embedding, Embedding, V],
        embedder: EdgeEmbedder.Params[V],
        head: EdgeHead.Params[V],
        queries: Tensor2[PoolQuery, Embedding, V],
        positions: LearnedAbsolutePositionalInjector.Params[Edge, Embedding, V]
    )

    object EdgeParams:

      given tensorTree: TensorTree[EdgeParams[Float32]] = TensorTree.derived
      given tree: TreeOf[EdgeParams[Float32], Float32] = TreeOf.derived

    /** @param nodes  How many nodes of a record the model can hold. One more than the most any
      *               record of the data draws, so that the last prediction embedding has somewhere
      *               to say the nodes have ended.
      * @param edges  The same for the relationships between them.
      * @param queries How many query vectors a slot may be asked with — the pool.
      * @param canvas The width of the drawing, which is how many pixels a coordinate chooses from.
      */
    def init(
        numLayers: Int,
        numHeads: Int,
        embedding: Int,
        nodes: Int,
        edges: Int,
        queries: Int,
        canvas: Int,
        key: Key
    ): Params[Float32] =
      val (setKey, edgeDecoderKey, edgeEmbedderKey, edgeHeadKey, edgeTokenKey, edgePositionKey) = key.splitToTuple(6)

      val embeddingExtent = Axis[Embedding] -> embedding
      val embeddingMixedExtent = Axis[EmbeddingMixed] -> embedding * 4
      val relationshipExtent = Axis[FloorplanRelationships] -> FloorplanRelationship.values.length
      val edgeExtent = Axis[Edge] -> edges

      val (subjectKey, objKey, relationshipKey) = edgeEmbedderKey.splitToTuple(3)
      val (relationshipHeadKey, subjectHeadKey, objHeadKey) = edgeHeadKey.splitToTuple(3)

      Params(
        set = D2S.Params.init(numLayers, numHeads, embedding, nodes, queries, canvas, setKey),
        edges = EdgeParams(
          decoder = EdgeDecoder.Params.xavierUniformDepthScaled(numLayers, numHeads, embeddingExtent, embeddingExtent, embeddingMixedExtent, edgeDecoderKey),
          embedder = EdgeEmbedder.Params(
            relationship = VocabularyEmbedder.Params.init(relationshipExtent, embeddingExtent, relationshipKey),
            subject = AffineLayer.Params.init(embeddingExtent, embeddingExtent, subjectKey),
            obj = AffineLayer.Params.init(embeddingExtent, embeddingExtent, objKey)
          ),
          head = EdgeHead.Params(
            relationship = AffineLayer.Params.init(embeddingExtent, relationshipExtent, relationshipHeadKey),
            subject = Pointer.Params.init(embeddingExtent, subjectHeadKey),
            obj = Pointer.Params.init(embeddingExtent, objHeadKey)
          ),
          queries = Init.xavierUniform(Axis[PoolQuery] -> queries, embeddingExtent, edgeTokenKey),
          positions = LearnedAbsolutePositionalInjector.Params.lecunNormal(edgeExtent, embeddingExtent, edgePositionKey)
        )
      )
