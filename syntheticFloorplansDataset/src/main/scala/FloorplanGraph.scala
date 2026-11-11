package syntheticFloorplansDataset

import dataset.DrawingDataset.Split
import dataset.Point
import dataset.PythonModules
import dataset.RecordScoring
import dataset.Runs
import dataset.Tolerances
import dimwit.*
import dimwit.jax.Jax
import dimwit.python.PyBridge.liftPyTensor
import dimwit.python.PyBridge.toPyTensor
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.SeqConverters

import java.nio.file.Files
import java.nio.file.Path

/** One node of a floor plan, as the host holds it: its class, and the two ends of its segment as a
  * fraction of the drawing's side.
  */
case class FloorplanNode(nodeClass: FloorplanClass, start: Point, end: Point)

/** One relationship of a floor plan, by the index of the [[FloorplanNode]]s it links. */
case class FloorplanEdge(relationship: FloorplanRelationship, subject: Int, obj: Int)

/** A floor plan's record with nothing in it in any particular order, which is what a record is: what
  * a transcription is scored as and written down as.
  */
case class FloorplanGraph(nodes: Seq[FloorplanNode], edges: Seq[FloorplanEdge]):

  /** The record laid out along the node and the edge axis, in its own order, the positions it does
    * not reach holding nothing.
    */
  def record[Node: Label, Edge: Label](nodeSlots: AxisExtent[Node], edgeSlots: AxisExtent[Edge]): FloorplanRecord[Node, Edge] =
    require(nodes.size <= nodeSlots.size, s"a record of ${nodes.size} nodes does not fit in ${nodeSlots.size}")
    require(edges.size <= edgeSlots.size, s"a record of ${edges.size} relationships does not fit in ${edgeSlots.size}")
    def placed(of: FloorplanNode => Float) = Tensor1(nodeSlots.axis, VType[Float32]).fromArray(Array.tabulate(nodeSlots.size)(at => nodes.lift(at).fold(0f)(of)))
    def named(of: FloorplanEdge => Int) = Tensor1(edgeSlots.axis, VType[Int32]).fromArray(Array.tabulate(edgeSlots.size)(at => edges.lift(at).fold(0)(of)))
    FloorplanRecord(
      FloorplanNodes(
        Tensor1(nodeSlots.axis, VType[Int32]).fromArray(Array.tabulate(nodeSlots.size)(at => nodes.lift(at).fold(FloorplanClass.NoNode)(_.nodeClass).id)),
        placed(_.start.x),
        placed(_.start.y),
        placed(_.end.x),
        placed(_.end.y)
      ),
      FloorplanEdges(named(_.relationship.id), named(_.subject), named(_.obj))
    )

  /** How many of every class and relationship the plan holds, e.g. `12 outside walls, 3 doors, 14 joined`. */
  def counted: String =
    def times[T](kinds: Seq[T], name: T => String, of: T => Int) =
      kinds.flatMap(kind => Option.when(of(kind) > 0)(s"${of(kind)} ${name(kind)}"))
    val held =
      times(FloorplanClass.values.toSeq.filter(_.isNode), nodeClass => s"${nodeClass.recordName}s", nodeClass => nodes.count(_.nodeClass == nodeClass)) ++
        times(FloorplanRelationship.values.toSeq.filter(_.isRelationship), _.recordName, related => edges.count(_.relationship == related))
    if held.isEmpty then "nothing" else held.mkString(", ")

  /** The fields of a record's JSON object: its nodes, their points in the pixels the drawings are stored at, 
    * and its relationships, naming nodes by their position.
    */
  def recordFields: String =
    val side = SyntheticFloorplans.storedSide
    def point(at: Point) = f"[${at.x * side}%.2f, ${at.y * side}%.2f]"
    def node(at: FloorplanNode) = s"""{"class": "${at.nodeClass.recordName}", "points": [${point(at.start)}, ${point(at.end)}]}"""
    def edge(at: FloorplanEdge) = s"""{"type": "${at.relationship.recordName}", "nodes": [${at.subject}, ${at.obj}]}"""
    s""""nodes": ${nodes.map(node).mkString("[", ", ", "]")}, "relationships": ${edges.map(edge).mkString("[", ", ", "]")}"""

object FloorplanGraph:

  /** The record a layout holds, every relationship resolved to the nodes it links. One naming a
    * position that holds no node is dropped: it relates nothing.
    */
  def of[Node: Label, Edge: Label](record: FloorplanRecord[Node, Edge]): FloorplanGraph =
    val nodes = record.nodes
    read(nodes.nodeClass.toArray, nodes.startX.toArray, nodes.startY.toArray, nodes.endX.toArray, nodes.endY.toArray,
      record.edges.relationship.toArray, record.edges.subject.toArray, record.edges.obj.toArray)

  /** The nodes of a layout alone, as a plan holding no relationships. */
  def of[Node: Label](nodes: FloorplanNodes[Node]): FloorplanGraph =
    read(nodes.nodeClass.toArray, nodes.startX.toArray, nodes.startY.toArray, nodes.endX.toArray, nodes.endY.toArray, Array.empty, Array.empty, Array.empty)

  /** Every record of a batch, read to the host — once for the batch, not once per plan. */
  def of[S: Label, Node: Label, Edge: Label](records: FloorplanRecordBatch[S, Node, Edge]): Seq[FloorplanGraph] =
    val (nodeClass, startX, startY, endX, endY) = (records.nodeClass.toArray, records.startX.toArray, records.startY.toArray, records.endX.toArray, records.endY.toArray)
    // DimWit's `toArray` returns no rows at all for a tensor whose second axis is empty.
    def perPlan(named: Tensor2[S, Edge, Int32]) =
      if named.shape(Axis[Edge]) == 0 then Array.fill(nodeClass.length)(Array.empty[Int]) else named.toArray
    val (relationship, subject, obj) = (perPlan(records.relationship), perPlan(records.subject), perPlan(records.obj))
    nodeClass.indices.map(plan => read(nodeClass(plan), startX(plan), startY(plan), endX(plan), endY(plan), relationship(plan), subject(plan), obj(plan)))

  private def read(nodeClass: Array[Int], startX: Array[Float], startY: Array[Float], endX: Array[Float], endY: Array[Float],
                   relationship: Array[Int], subject: Array[Int], obj: Array[Int]): FloorplanGraph =
    val classes = nodeClass.map(FloorplanClass.fromId)
    val nodeAt = classes.indices.filter(classes(_).isNode).zipWithIndex.toMap
    FloorplanGraph(
      nodes = nodeAt.keys.toSeq.sorted.map(at => FloorplanNode(classes(at), Point(startX(at), startY(at)), Point(endX(at), endY(at)))),
      edges = relationship.indices.map(at => (FloorplanRelationship.fromId(relationship(at)), at)).filter(_._1.isRelationship).flatMap: (related, at) =>
        for
          from <- nodeAt.get(subject(at))
          to <- nodeAt.get(obj(at))
        yield FloorplanEdge(related, from, to)
    )

/** Scoring a transcribed floor plan against the record its drawing was drawn from, as
  * [[RecordScoring]] scores the other corpora and in its terms: nodes are matched by what they say —
  * their class and both ends within the tolerance — and relationships by the nodes they name.
  */
object FloorplanScoring:

  def score(target: FloorplanGraph, found: FloorplanGraph, tolerance: Float): RecordScoring.Scored =
    val matched = matching(target.nodes, found.nodes, tolerance)
    def multiset(edges: Seq[FloorplanEdge]) = edges.groupBy(identity).view.mapValues(_.length).toMap
    val wanted = multiset(target.edges.map(canonical))
    val present = multiset(found.edges.flatMap(resolve(matched)).map(canonical))
    val nodesExact = matched.size == target.nodes.length && found.nodes.length == target.nodes.length
    RecordScoring.Scored(
      nodes = target.nodes.length,
      nodesFound = matched.size,
      nodesPredicted = found.nodes.length,
      nodesExact = nodesExact,
      relationships = target.edges.length,
      relationshipsFound = wanted.map((edge, count) => math.min(count, present.getOrElse(edge, 0))).sum,
      relationshipsPredicted = found.edges.length,
      isExact = nodesExact && found.edges.length == present.values.sum && present == wanted
    )

  /** Which found node stands for which target node, by slot. Greedy, as [[RecordScoring.matching]]. */
  def matching(target: Seq[FloorplanNode], found: Seq[FloorplanNode], tolerance: Float): Map[Int, Int] =
    target.zipWithIndex.foldLeft(Map.empty[Int, Int]):
      case (matched, (node, index)) =>
        found.indices
          .find(slot => !matched.contains(slot) && isFound(node, found(slot), tolerance))
          .fold(matched)(slot => matched + (slot -> index))

  private def isFound(target: FloorplanNode, found: FloorplanNode, tolerance: Float): Boolean =
    def near(wanted: Point, at: Point) = (wanted.x - at.x).abs <= tolerance && (wanted.y - at.y).abs <= tolerance
    target.nodeClass == found.nodeClass && near(target.start, found.start) && near(target.end, found.end)

  private def resolve(matched: Map[Int, Int])(edge: FloorplanEdge): Option[FloorplanEdge] =
    for
      subject <- matched.get(edge.subject)
      obj <- matched.get(edge.obj)
    yield FloorplanEdge(edge.relationship, subject, obj)

  /** A symmetric relationship says nothing by which of its two nodes comes first. */
  private def canonical(edge: FloorplanEdge): FloorplanEdge =
    if edge.relationship.isSymmetric && edge.subject > edge.obj then FloorplanEdge(edge.relationship, edge.obj, edge.subject) else edge

/** What a model wrote down for every floor plan of a split, beside what the plan holds, in the
  * dataset's own form: `<model>-synthetic-floorplans-<size>-<split>-<step>.jsonl` in
  * [[Runs.outputDir]], one plan per line, in the order of the split.
  *
  * {{{
  * {"plan": 0,
  *  "target":    {"nodes": [{"class": "outside wall", "points": [[x, y], [x, y]]}, ...], "relationships": [{"type": "in", "nodes": [14, 1]}, ...]},
  *  "predicted": {"nodes": [...], "relationships": [...]},
  *  "matched":   {"0": [null, ...], "2": [1, ...], "4": [1, ...], "8": [1, ...]}}
  * }}}
  *
  * Points are in the pixels the drawings are stored at, as in the dataset. The predicted nodes are in
  * the order the model wrote them down. `matched` holds, at every tolerance in pixels of the
  * `canvas` the model read, the target node each predicted node stands for, or `null`.
  */
object FloorplanTranscripts:

  def write(model: String, size: String, split: Split, step: Int, canvas: Int, plans: Seq[(FloorplanGraph, FloorplanGraph)]): Path =
    def record(at: FloorplanGraph) = s"{${at.recordFields}}"

    val lines = plans.zipWithIndex.map:
      case ((target, predicted), plan) =>
        val matched = Tolerances.map: tolerance =>
          val standsFor = FloorplanScoring.matching(target.nodes, predicted.nodes, tolerance / canvas)
          s""""${tolerance.toInt}": ${predicted.nodes.indices.map(standsFor.get(_).fold("null")(_.toString)).mkString("[", ", ", "]")}"""
        s"""{"plan": $plan, "target": ${record(target)}, "predicted": ${record(predicted)}, "matched": ${matched.mkString("{", ", ", "}")}}"""

    val path = Path.of(Runs.outputDir, s"$model-${SyntheticFloorplans.name}-$size-${split.fileName}-$step.jsonl")
    Files.createDirectories(path.getParent)
    Files.writeString(path, lines.mkString("", "\n", "\n"))

/** Draws a floor plan's record as the picture it stands for, over a drawing: see
  * `floorplan_drawing.py`, which is where the drawing happens.
  */
object FloorplanDrawing:

  def apply[W: Label, H: Label, C: Label](plan: FloorplanGraph, over: Tensor2[W, H, UInt8], colour: Axis[C]): Tensor3[W, H, C, UInt8] =
    val nodes = plan.nodes
    def at(point: FloorplanNode => Float) = nodes.map(point).toPythonCopy
    liftPyTensor[(W, H, C), UInt8](
      Jax.jnp.asarray(
        module.render(
          toPyTensor(over),
          nodes.map(_.nodeClass.id).toPythonCopy,
          at(_.start.x),
          at(_.start.y),
          at(_.end.x),
          at(_.end.y),
          plan.edges.map(_.relationship.id).toPythonCopy,
          plan.edges.map(_.subject).toPythonCopy,
          plan.edges.map(_.obj).toPythonCopy,
          FloorplanClass.values.map(_.recordName).toSeq.toPythonCopy,
          FloorplanRelationship.values.map(_.recordName).toSeq.toPythonCopy
        )
      )
    )

  private lazy val module: py.Dynamic = PythonModules("floorplan_drawing", "./syntheticFloorplansDataset/src/main/resources/python")
