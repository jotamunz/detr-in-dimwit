package d2s.sketch.model

import d2s.*
import d2s.sketch.*
import d2s.model.*
import dataset.NodeClass
import dataset.NodeClasses
import dataset.IsConstruction
import dataset.RecordNodes
import deepwit.attention.KVCache
import deepwit.base.AffineLayer
import documentEncoder.DocumentEncoder
import deepwit.embedder.LearnedAbsolutePositionalInjector
import deepwit.embedder.VocabularyEmbedder
import deepwit.init.Init
import NodeEmbedder.NodePart
import NodeEmbedder.PartEmbedding
import NodeHead.NodeLogits
import dimwit.*

import scala.language.implicitConversions

/** Document-to-set model based on remaining-node prediction.
  *
  * 1. The document (or: image) is embedded by a vision transformer to a sequence of patch embeddings.
  * 2. The nodes are predicted based on cross-attenting the document (1).
  */
class D2S[V: IsFloating](params: D2S.Params[V]):

  import D2S.NodeQueryLogits

  val encodeDocument = DocumentEncoder(params.encoder)

  private val embedNodes = NodeEmbedder(params.nodes.embedder)
  private val nodePosition = LearnedAbsolutePositionalInjector(params.nodes.positions)
  private val nodeDecoder = NodeDecoder(params.nodes.decoder)
  val nodeHead = NodeHead(params.nodes.head)

  val pool = params.nodes.queries.shape(Axis[PoolQuery])

  /** What two queries of the pool answer. Queries selected randomly. */
  def logits(document: Tensor3[Width, Height, Channel, V], taken: RecordNodes[Node], asked: Key): NodeQueryLogits[V] =
    val randomQueryIds = Random.permutation(Axis[PoolQuery] -> pool)(asked).slice(Axis[PoolQuery].at(0 until 2))
    predict(encodeDocument(document), taken, randomQueryIds)._2

  /** What every query of the pool answers, in one reading. */
  def logitsPerQuery(encoded: Tensor2[Patch, Embedding, V], taken: RecordNodes[Node]): NodeQueryLogits[V] =
    val allQueryIds = Tensor1(Axis[PoolQuery], VType[Int32]).fromArray(Array.range(0, pool))
    predict(encoded, taken, allQueryIds)._2

  /** The taken nodes as the decoder carries them, and what each asked query answers at every slot. */
  def predict(
      encodedDocument: Tensor2[Patch, Embedding, V],
      taken: RecordNodes[Node],
      nodeQueryIds: Tensor1[PoolQuery, Int32]
  ): (Tensor2[Node, Embedding, V], NodeQueryLogits[V]) =

    val (carriedNodes, answeredNodes) =
      val takenNodes = nodePosition(embedNodes(taken))
      val queryNodes = params.nodes.queries.slice(Axis[PoolQuery].at(nodeQueryIds)) // take queries and broadcast along context
        .vmap(Axis[PoolQuery]): query =>
          nodePosition(query.broadcastTo(takenNodes.shape))
      nodeDecoder.forTraining(encodedDocument, takenNodes, queryNodes)

    val (nodeClass, construction, startX, startY, endX, endY, midX, midY) =
      answeredNodes.vmap(Axis[PoolQuery]): answered =>
        val scored = nodeHead(answered)
        (scored.nodeClass, scored.construction, scored.startX, scored.startY, scored.endX, scored.endY, scored.midX, scored.midY)

    (carriedNodes, NodeQueryLogits(nodeClass, construction, startX, startY, endX, endY, midX, midY))

  // Writing a record down one node at a time, as [[predict]] would have it at every slot, but
  // projecting at each step only what the step adds.

  /** The encoded document as every step of the decoder reads it. */
  def read(encodedDocument: Tensor2[Patch, Embedding, V]): List[KVCache[Patch, V]] =
    nodeDecoder.read(encodedDocument)

  /** The decoder before any node is taken. */
  def nothingTaken(slots: AxisExtent[Node]): List[KVCache[Node, V]] =
    nodeDecoder.nothingTaken(slots)

  /** Takes `node`, a record of one node, at `slot`, for the slots after it to read: the node as the
    * decoder carries it, and the taken nodes with it among them.
    */
  def take(documentCache: List[KVCache[Patch, V]], taken: List[KVCache[Node, V]], slot: Tensor0[Int32], node: RecordNodes[Node]): (Tensor1[Embedding, V], List[KVCache[Node, V]]) =
    val embedded = nodePosition.injectAt(slot)(embedNodes(node).squeeze(Axis[Node]))
    nodeDecoder.take(documentCache, taken, slot, embedded)

  /** What the pool answers at `slot`, given the nodes taken before it: a candidate node per query,
    * in the order of the queries.
    */
  def answerAt(documentCache: List[KVCache[Patch, V]], taken: List[KVCache[Node, V]], slot: Tensor0[Int32]): NodeLogits[V] =
    val asked = params.nodes.queries.vmap(Axis[PoolQuery])(nodePosition.injectAt(slot))
    nodeHead(nodeDecoder.answer(documentCache, taken, slot, asked).relabel(Axis[PoolQuery] -> Axis[Node]))

object D2S:

  /** [[NodeLogits]] at every query slot. */
  case class NodeQueryLogits[V](
      nodeClass: Tensor3[PoolQuery, Node, NodeClasses, V],
      construction: Tensor3[PoolQuery, Node, IsConstruction, V],
      startX: Tensor3[PoolQuery, Node, Pixel, V],
      startY: Tensor3[PoolQuery, Node, Pixel, V],
      endX: Tensor3[PoolQuery, Node, Pixel, V],
      endY: Tensor3[PoolQuery, Node, Pixel, V],
      midX: Tensor3[PoolQuery, Node, Pixel, V],
      midY: Tensor3[PoolQuery, Node, Pixel, V]
  ):
    def at(query: Int): NodeLogits[V] = NodeLogits(
      nodeClass.slice(Axis[PoolQuery].at(query)),
      construction.slice(Axis[PoolQuery].at(query)),
      startX.slice(Axis[PoolQuery].at(query)),
      startY.slice(Axis[PoolQuery].at(query)),
      endX.slice(Axis[PoolQuery].at(query)),
      endY.slice(Axis[PoolQuery].at(query)),
      midX.slice(Axis[PoolQuery].at(query)),
      midY.slice(Axis[PoolQuery].at(query))
    )

  object NodeQueryLogits:

    given tensorTree[V]: TensorTree[NodeQueryLogits[V]] = TensorTree.derived
    given tree[V]: TreeOf[NodeQueryLogits[V], V] = TreeOf.derived

  case class Params[V](
      encoder: DocumentEncoder.Params[Embedding, V],
      nodes: Params.NodeParams[V]
  )

  object Params:

    given tensorTree: TensorTree[Params[Float32]] = TensorTree.derived
    given tree: TreeOf[Params[Float32], Float32] = TreeOf.derived

    case class NodeParams[V](
        decoder: NodeDecoder.Params[Embedding, Embedding, V],
        embedder: NodeEmbedder.Params[V],
        head: NodeHead.Params[V],
        queries: Tensor2[PoolQuery, Embedding, V],
        positions: LearnedAbsolutePositionalInjector.Params[Node, Embedding, V]
    )

    object NodeParams:

      given tensorTree: TensorTree[NodeParams[Float32]] = TensorTree.derived
      given tree: TreeOf[NodeParams[Float32], Float32] = TreeOf.derived

    /** @param nodes  How many nodes of a record the model can hold. One more than the most any
      *               record of the data draws, so that the last prediction embedding has somewhere
      *               to say the nodes have ended.
      * @param queries How many query vectors a slot may be asked with — the pool.
      * @param canvas The width of the drawing, which is how many pixels a coordinate chooses from.
      */
    def init(
        numLayers: Int,
        numHeads: Int,
        embedding: Int,
        nodes: Int,
        queries: Int,
        canvas: Int,
        key: Key
    ): Params[Float32] =
      val (encoderKey, nodeDecoderKey, nodeEmbedderKey, nodeHeadKey, nodeTokenKey, nodePositionKey) = key.splitToTuple(6)

      val embeddingExtent = Axis[Embedding] -> embedding
      val embeddingMixedExtent = Axis[EmbeddingMixed] -> embedding * 4
      val partExtent = Axis[PartEmbedding] -> embedding / 8
      val nodeClassExtent = Axis[dataset.NodeClasses] -> NodeClass.values.length
      val constructionExtent = Axis[IsConstruction] -> 2
      val pixelExtent = Axis[Pixel] -> canvas
      val nodeExtent = Axis[Node] -> nodes
      // A node embedding is put together from its class, whether it is construction geometry, and
      // the six coordinates a class can place.
      val nodePartExtent = Axis[NodePart |*| PartEmbedding] -> 8 * partExtent.size

      val (startXKey, startYKey, endXKey, endYKey, midXKey, midYKey, nodeClassKey, constructionKey, nodeProjectionKey) = nodeEmbedderKey.splitToTuple(9)
      val (classHeadKey, constructionHeadKey, startXHeadKey, startYHeadKey, endXHeadKey, endYHeadKey, midXHeadKey, midYHeadKey) = nodeHeadKey.splitToTuple(8)

      Params(
        encoder = DocumentEncoder.Params.xavierUniformDepthScaled(numLayers, numHeads, embeddingExtent, Axis[DocumentEncoder.EmbeddingMixed] -> embeddingMixedExtent.size, encoderKey),
        nodes = NodeParams(
          decoder = NodeDecoder.Params.xavierUniformDepthScaled(numLayers, numHeads, embeddingExtent, embeddingExtent, embeddingMixedExtent, nodeDecoderKey),
          embedder = NodeEmbedder.Params(
            nodeClass = VocabularyEmbedder.Params.init(nodeClassExtent, partExtent, nodeClassKey),
            construction = VocabularyEmbedder.Params.init(constructionExtent, partExtent, constructionKey),
            startX = VocabularyEmbedder.Params.init(pixelExtent, partExtent, startXKey),
            startY = VocabularyEmbedder.Params.init(pixelExtent, partExtent, startYKey),
            endX = VocabularyEmbedder.Params.init(pixelExtent, partExtent, endXKey),
            endY = VocabularyEmbedder.Params.init(pixelExtent, partExtent, endYKey),
            midX = VocabularyEmbedder.Params.init(pixelExtent, partExtent, midXKey),
            midY = VocabularyEmbedder.Params.init(pixelExtent, partExtent, midYKey),
            projection = AffineLayer.Params.init(nodePartExtent, embeddingExtent, nodeProjectionKey)
          ),
          head = NodeHead.Params(
            nodeClass = AffineLayer.Params.init(embeddingExtent, nodeClassExtent, classHeadKey),
            construction = AffineLayer.Params.init(embeddingExtent, constructionExtent, constructionHeadKey),
            startX = AffineLayer.Params.init(embeddingExtent, pixelExtent, startXHeadKey),
            startY = AffineLayer.Params.init(embeddingExtent, pixelExtent, startYHeadKey),
            endX = AffineLayer.Params.init(embeddingExtent, pixelExtent, endXHeadKey),
            endY = AffineLayer.Params.init(embeddingExtent, pixelExtent, endYHeadKey),
            midX = AffineLayer.Params.init(embeddingExtent, pixelExtent, midXHeadKey),
            midY = AffineLayer.Params.init(embeddingExtent, pixelExtent, midYHeadKey)
          ),
          queries = Init.xavierUniform(Axis[PoolQuery] -> queries, embeddingExtent, nodeTokenKey),
          positions = LearnedAbsolutePositionalInjector.Params.lecunNormal(nodeExtent, embeddingExtent, nodePositionKey)
        )
      )
