import d2g.*
import d2g.model.*
import d2g.floorplan.eval.Transcriber
import d2g.floorplan.model.*
import d2g.floorplan.model.EdgeHead.EdgeLogits
import d2s.eval.chosen
import d2s.model.PoolQuery
import dimwit.*
import dimwit.stats.Uniform
import syntheticFloorplansDataset.FloorplanClass
import syntheticFloorplansDataset.FloorplanClasses
import syntheticFloorplansDataset.FloorplanEdges
import syntheticFloorplansDataset.FloorplanGraph
import syntheticFloorplansDataset.FloorplanNodes
import syntheticFloorplansDataset.FloorplanRecord
import syntheticFloorplansDataset.FloorplanRelationship
import syntheticFloorplansDataset.FloorplanRelationships

class CachedEdgeDecodingSuite extends munit.FunSuite:

  override def beforeAll(): Unit = dimwit.initialize()

  test("decoding one relationship at a time with caches answers as the whole record read at once"):
    val canvas = 64
    val (nodeSlots, edgeSlots) = (5, 4)
    val params = D2G.Params.init(numLayers = 2, numHeads = 2, embedding = 32, nodes = nodeSlots, edges = edgeSlots, queries = 3, canvas = canvas, key = Random.Key(0))
    val model = D2G(params)
    val pixels = Shape3(Axis[Width] -> canvas, Axis[Height] -> canvas, Axis[Channel] -> 1)
    val drawing = Uniform(Tensor(pixels).fill(0f), Tensor(pixels).fill(1f)).sample(Random.Key(1))
    def field(values: Float*) = Tensor1(Axis[Node], VType[Float32]).fromArray(values.toArray)
    val nodes = FloorplanNodes(
      nodeClass = Tensor1(Axis[Node], VType[Int32]).fromArray(Array(FloorplanClass.OutsideWall, FloorplanClass.OutsideWall, FloorplanClass.InsideWall, FloorplanClass.Window, FloorplanClass.NoNode).map(_.id)),
      startX = field(0.1f, 0.1f, 0.5f, 0.1f, 0f),
      startY = field(0.1f, 0.1f, 0.1f, 0.3f, 0f),
      endX = field(0.9f, 0.1f, 0.5f, 0.1f, 0f),
      endY = field(0.1f, 0.9f, 0.9f, 0.5f, 0f)
    )
    def ends(values: Int*) = Tensor1(Axis[Edge], VType[Int32]).fromArray(values.toArray)
    val edges = FloorplanEdges(
      relationship = Tensor1(Axis[Edge], VType[Int32]).fromArray(Array(FloorplanRelationship.Joined, FloorplanRelationship.In, FloorplanRelationship.Joined, FloorplanRelationship.NoRelationship).map(_.id)),
      subject = ends(0, 3, 0, 0),
      obj = ends(1, 1, 2, 0)
    )
    val encoded = model.encodeDocument(drawing)
    val whole = model.logitsPerQuery(encoded, FloorplanRecord(nodes, edges)).edges

    def oneNode(slot: Int) = FloorplanNodes(nodes.nodeClass.slice(Axis[Node].at(slot, 1)), nodes.startX.slice(Axis[Node].at(slot, 1)),
      nodes.startY.slice(Axis[Node].at(slot, 1)), nodes.endX.slice(Axis[Node].at(slot, 1)), nodes.endY.slice(Axis[Node].at(slot, 1)))
    val documentCache = model.set.read(encoded)
    val (carried, _) = (0 until nodeSlots).foldLeft((Seq.empty[Tensor1[Embedding, Float32]], model.set.nothingTaken(Axis[Node] -> nodeSlots))):
      case ((carried, taken), slot) =>
        val (carriedNode, nowTaken) = model.set.take(documentCache, taken, slot, oneNode(slot))
        (carried :+ carriedNode, nowTaken)
    val written = NodeSource(stack(carried, Axis[Node]), !(nodes.nodeClass elementEquals_! FloorplanClass.NoNode.id))
    val sources = model.readForEdges(encoded, written)

    def oneEdge(values: Tensor1[Edge, Int32], slot: Int) = values.slice(Axis[Edge].at(slot, 1))
    (0 until edgeSlots).foldLeft(model.noEdgeTaken(Axis[Edge] -> edgeSlots)): (taken, slot) =>
      val (_, atSlot) = model.answerEdgeAt(written, sources, taken, slot)
      def assertSame[L: Label](name: String, cached: Tensor2[Edge, L, Float32], read: Tensor3[PoolQuery, Edge, L, Float32]) =
        val difference = (cached.relabel(Axis[Edge] -> Axis[PoolQuery]) - read.slice(Axis[Edge].at(slot))).abs.max.item
        assert(difference < 1e-4f, s"slot $slot, $name: the cached answer is off by $difference")
      assertSame("relationship", atSlot.relationship, whole.relationship)
      assertSame("subject", atSlot.subject, whole.subject)
      assertSame("object", atSlot.obj, whole.obj)
      model.takeEdge(written, sources, taken, slot, FloorplanEdges(oneEdge(edges.relationship, slot), oneEdge(edges.subject, slot), oneEdge(edges.obj, slot)))

  test("the transcriber writes the nodes d2s would, and the relationships reading the whole record at every slot would"):
    val canvas = 64
    val (nodes, edges) = (Axis[Node] -> 5, Axis[Edge] -> 4)
    val untrained = D2G.Params.init(numLayers = 2, numHeads = 2, embedding = 32, nodes = nodes.size, edges = edges.size, queries = 3, canvas = canvas, key = Random.Key(2))
    // An untrained model stops at once; one that never says it is done writes every slot.
    val nodeClassHead = untrained.set.nodes.head.nodeClass
    val relationshipHead = untrained.edges.head.relationship
    val params = untrained.copy(
      set = untrained.set.copy(nodes = untrained.set.nodes.copy(head = untrained.set.nodes.head.copy(nodeClass =
        nodeClassHead.copy(bias = nodeClassHead.bias.set(Axis[FloorplanClasses].at(FloorplanClass.NoNode.id))(-100f))))),
      edges = untrained.edges.copy(head = untrained.edges.head.copy(relationship =
        relationshipHead.copy(bias = relationshipHead.bias.set(Axis[FloorplanRelationships].at(FloorplanRelationship.NoRelationship.id))(-100f))))
    )
    val model = D2G(params)
    val pixels = Shape3(Axis[Width] -> canvas, Axis[Height] -> canvas, Axis[Channel] -> 1)
    val documents = (3 to 5).map(seed => Uniform(Tensor(pixels).fill(0f), Tensor(pixels).fill(1f)).sample(Random.Key(seed)))
    val readWhole = jit((document: Tensor3[Width, Height, Channel, Float32], taken: FloorplanRecord[Node, Edge]) => model.logitsPerQuery(model.encodeDocument(document), taken).edges)

    /** The relationships of `written`, read again in whole at every slot, and kept as the model wrote them. */
    def readAgainAtEverySlot(document: Tensor3[Width, Height, Channel, Float32], written: FloorplanNodes[Node]): FloorplanEdges[Edge] =
      def filled(value: Int) = Tensor1(edges.axis, VType[Int32]).fromArray(Array.fill(edges.size)(value))
      val nothing = FloorplanEdges(filled(FloorplanRelationship.NoRelationship.id), filled(0), filled(0))
      (0 until edges.size).foldLeft((nothing, true)):
        case ((taken, writing), slot) if !writing => (taken, false)
        case ((taken, _), slot) =>
          val scored = readWhole(document, FloorplanRecord(written, taken))
          def candidates[L: Label](logits: Tensor3[PoolQuery, Edge, L, Float32]) = logits.slice(Axis[Edge].at(slot)).relabel(Axis[PoolQuery] -> Axis[Edge])
          val logits = EdgeLogits(candidates(scored.relationship), candidates(scored.subject), candidates(scored.obj))
          val said = model.edgeHead.decide(logits)
          val likeliest = Axis[Edge].at((chosen(logits.relationship) + chosen(logits.subject) + chosen(logits.obj)).argmax(Axis[Edge]).item)
          def put(values: Tensor1[Edge, Int32], from: Tensor1[Edge, Int32]) = values.set(Axis[Edge].at(slot))(from.slice(likeliest))
          if said.relationship.slice(likeliest).item == FloorplanRelationship.NoRelationship.id then (taken, false)
          else (FloorplanEdges(put(taken.relationship, said.relationship), put(taken.subject, said.subject), put(taken.obj, said.obj)), true)
      ._1

    val transcribed = Transcriber(nodes, edges, drawings = documents.size)(params, documents)
    val setTranscribed = d2s.floorplan.eval.Transcriber(nodes, drawings = documents.size)(params.set, documents)
    documents.lazyZip(transcribed).lazyZip(setTranscribed).zipWithIndex.foreach:
      case ((document, written, writtenBySet), at) =>
        assertEquals(written.nodes, writtenBySet.nodes, s"plan $at")
        assertEquals(written.edges.size, edges.size, s"plan $at: not a relationship in every slot")
        val writtenNodes = FloorplanGraph(written.nodes, Seq.empty).record(nodes, edges).nodes
        assertEquals(written.edges, FloorplanGraph.of(FloorplanRecord(writtenNodes, readAgainAtEverySlot(document, writtenNodes))).edges, s"plan $at")
