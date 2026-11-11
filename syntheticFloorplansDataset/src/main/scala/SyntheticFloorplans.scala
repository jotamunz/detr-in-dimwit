package syntheticFloorplansDataset

import dataset.Batch
import dataset.DrawingDataset.Split
import dataset.PythonModules
import dataset.Sample
import dataset.drawingsOf
import dimwit.*
import dimwit.jax.Jax
import dimwit.python.PyBridge.liftPyTensor
import dimwit.tensor.Tensor4
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.SeqConverters

import scala.language.implicitConversions

/** Axis over the [[FloorplanClass]] values a node of a floor plan is classified into. */
trait FloorplanClasses derives Label

/** Axis over the [[FloorplanRelationship]] values a relationship of a floor plan is classified into. */
trait FloorplanRelationships derives Label

/** What a node of a floor plan is: a wall, a door or a window, each a horizontal or vertical segment
  * placed by its two ends, read rightwards or downwards.
  *
  * [[NoNode]] marks a position the record does not reach, which is also where a transcription of its
  * nodes stops.
  */
enum FloorplanClass(val id: Int, val recordName: String):

  case NoNode extends FloorplanClass(0, "")

  /** A wall with the outside on one side, from a corner or junction to the next. */
  case OutsideWall extends FloorplanClass(1, "outside wall")

  /** A wall with rooms on both sides, from a corner or junction to the next. */
  case InsideWall extends FloorplanClass(2, "inside wall")

  /** Placed by its jambs, in an inside wall. */
  case Door extends FloorplanClass(3, "door")

  /** Placed by its jambs, in an outside wall. */
  case OutsideDoor extends FloorplanClass(4, "outside door")

  /** Placed by its jambs, in an outside wall. */
  case Window extends FloorplanClass(5, "window")

  def isNode: Boolean = this != FloorplanClass.NoNode

object FloorplanClass:

  def fromId(id: Int): FloorplanClass =
    values.find(_.id == id).getOrElse(throw IllegalArgumentException(s"unknown floor plan class id: $id"))

  def indicator[V: IsFloating](vtype: VType[V])(holds: FloorplanClass => Boolean): Tensor1[FloorplanClasses, V] =
    Tensor1(Axis[FloorplanClasses], vtype).fromArray(values.map(nodeClass => if holds(nodeClass) then 1f else 0f))

/** What a relationship of a floor plan is.
  *
  * [[NoRelationship]] marks a position the record does not reach.
  */
enum FloorplanRelationship(val id: Int, val recordName: String, val isSymmetric: Boolean):

  case NoRelationship extends FloorplanRelationship(0, "", false)

  /** Two walls sharing an end, held once with the two it links in ascending order. */
  case Joined extends FloorplanRelationship(1, "joined", true)

  /** A door, outside door or window, and the wall it stands in. */
  case In extends FloorplanRelationship(2, "in", false)

  def isRelationship: Boolean = this != FloorplanRelationship.NoRelationship

object FloorplanRelationship:

  def fromId(id: Int): FloorplanRelationship =
    values.find(_.id == id).getOrElse(throw IllegalArgumentException(s"unknown floor plan relationship id: $id"))

  def indicator[V: IsFloating](vtype: VType[V])(holds: FloorplanRelationship => Boolean): Tensor1[FloorplanRelationships, V] =
    Tensor1(Axis[FloorplanRelationships], vtype).fromArray(values.map(relationship => if holds(relationship) then 1f else 0f))

/** The nodes of a floor plan, laid out along `Node`: each a class and the two ends of a segment, as a
  * fraction of the drawing's side. The positions a record does not reach hold [[FloorplanClass.NoNode]]
  * and are placed nowhere.
  */
case class FloorplanNodes[Node](
    nodeClass: Tensor1[Node, Int32],
    startX: Tensor1[Node, Float32],
    startY: Tensor1[Node, Float32],
    endX: Tensor1[Node, Float32],
    endY: Tensor1[Node, Float32]
)

/** The relationships between those nodes, laid out along `Edge`, naming the nodes they relate by
  * their position along `Node`. The positions a record does not reach hold
  * [[FloorplanRelationship.NoRelationship]].
  */
case class FloorplanEdges[Edge](
    relationship: Tensor1[Edge, Int32],
    subject: Tensor1[Edge, Int32],
    obj: Tensor1[Edge, Int32]
)

/** What a floor plan's drawing encodes: its nodes, and the relationships between them. */
case class FloorplanRecord[Node, Edge](nodes: FloorplanNodes[Node], edges: FloorplanEdges[Edge])

/** [[FloorplanRecord]] for a batch of floor plans along the axis `S`. A model that predicts no
  * relationships reads the nodes alone.
  */
case class FloorplanRecordBatch[S, Node, Edge](
    nodeClass: Tensor2[S, Node, Int32],
    startX: Tensor2[S, Node, Float32],
    startY: Tensor2[S, Node, Float32],
    endX: Tensor2[S, Node, Float32],
    endY: Tensor2[S, Node, Float32],
    relationship: Tensor2[S, Edge, Int32],
    subject: Tensor2[S, Edge, Int32],
    obj: Tensor2[S, Edge, Int32]
):

  /** The same records, laid out again in `nodeSlots` and `edgeSlots` positions: their nodes by the
    * rank of their class, their relationships by the rank of the class of the node they start from,
    * lowest first, each in a fresh random order within a rank, and the positions they do not reach
    * last. Drawn on the device.
    *
    * @param classRank Ranks [[FloorplanClass.NoNode]] above every class, so that the positions a
    *                  record does not reach come last. Ranking every other class the same lays the
    *                  nodes out in any order.
    * @param startRank Ranking every class the same lays the relationships out in any order.
    */
  def permuted(
      key: Key,
      nodeSlots: AxisExtent[Node],
      edgeSlots: AxisExtent[Edge],
      classRank: Tensor1[FloorplanClasses, Float32] = FloorplanClass.indicator(VType[Float32])(!_.isNode),
      startRank: Tensor1[FloorplanClasses, Float32] = FloorplanClass.indicator(VType[Float32])(_ => false)
  )(using Label[S], Label[Node], Label[Edge]): FloorplanRecordBatch[S, Node, Edge] =
    val plans = nodeClass.shape.extent(Axis[S])
    val (heldNodes, heldEdges) = (nodeClass.shape(Axis[Node]), relationship.shape(Axis[Edge]))
    require(heldNodes <= nodeSlots.size, s"records of $heldNodes nodes do not fit in ${nodeSlots.size}")
    require(heldEdges <= edgeSlots.size, s"records of $heldEdges relationships do not fit in ${edgeSlots.size}")
    val (emptyNodes, emptyEdges) = (Axis[Node] -> (nodeSlots.size - heldNodes), Axis[Edge] -> (edgeSlots.size - heldEdges))
    def nowhere(placed: Tensor2[S, Node, Float32]) = concatenate(placed, Tensor(Shape(plans, emptyNodes), VType[Float32]).fill(0f), Axis[Node])
    def nothing(named: Tensor2[S, Edge, Int32]) = concatenate(named, Tensor(Shape(plans, emptyEdges), VType[Int32]).fill(0), Axis[Edge])
    val paddedClass = concatenate(nodeClass, Tensor(Shape(plans, emptyNodes), VType[Int32]).fill(FloorplanClass.NoNode.id), Axis[Node])
    val paddedRelationship = concatenate(relationship, Tensor(Shape(plans, emptyEdges), VType[Int32]).fill(FloorplanRelationship.NoRelationship.id), Axis[Edge])

    val (forNodes, forEdges) = key.split2()
    val (nodeKeys, edgeKeys) = (forNodes.splitToTensor(plans), forEdges.splitToTensor(plans))
    val (classes, startXs, startYs, endXs, endYs, nodeOrders) =
      zipvmap(Axis[S])(paddedClass, nowhere(startX), nowhere(startY), nowhere(endX), nowhere(endY), nodeKeys):
        case (nodeClass, startX, startY, endX, endY, key) =>
          val order = FloorplanRecordBatch.inOrder(classRank.slice(Axis[FloorplanClasses].at(nodeClass)), key.item)
          def reordered[V](placed: Tensor1[Node, V]) = placed.slice(Axis[Node].at(order))
          (reordered(nodeClass), reordered(startX), reordered(startY), reordered(endX), reordered(endY), order)
    val (relationships, subjects, objs) =
      zipvmap(Axis[S])(paddedRelationship, nothing(subject), nothing(obj), paddedClass, nodeOrders, edgeKeys):
        case (relationship, subject, obj, nodeClass, nodeOrder, key) =>
          val notThere = FloorplanRelationship.indicator(VType[Float32])(!_.isRelationship).slice(Axis[FloorplanRelationships].at(relationship))
          val startsFrom = startRank.slice(Axis[FloorplanClasses].at(nodeClass.slice(Axis[Node].at(subject))))
          val order = FloorplanRecordBatch.inOrder(startsFrom + notThere *! (startRank.max + 1f), key.item)
          val related = relationship.slice(Axis[Edge].at(order))
          // A relationship names the nodes it relates by their position, and a node that sat at
          // `at` before sits at `renamed(at)` now.
          val renamed = nodeOrder.argsort(Axis[Node])
          def moved(named: Tensor1[Edge, Int32]) = renamed.slice(Axis[Node].at(named.slice(Axis[Edge].at(order))))
          val (movedSubject, movedObj) = (moved(subject), moved(obj))
          def is(holds: FloorplanRelationship => Boolean) =
            val marked = FloorplanRelationship.indicator(VType[Float32])(holds).slice(Axis[FloorplanRelationships].at(related))
            marked > Tensor.like(marked).fill(0f)
          // A symmetric relationship names the two it relates in ascending order, and a position
          // holding no relationship relates nothing.
          val none = Tensor.like(movedSubject).fill(0)
          (
            related,
            where(is(_.isRelationship), where(is(_.isSymmetric), minimum(movedSubject, movedObj), movedSubject), none),
            where(is(_.isRelationship), where(is(_.isSymmetric), maximum(movedSubject, movedObj), movedObj), none)
          )
    FloorplanRecordBatch(classes, startXs, startYs, endXs, endYs, relationships, subjects, objs)

object FloorplanRecordBatch:

  /** A batch of records laid out in order, along `batch`. */
  def of[S: Label, Node: Label, Edge: Label](records: Seq[FloorplanGraph], batch: Axis[S], nodes: AxisExtent[Node], edges: AxisExtent[Edge]): FloorplanRecordBatch[S, Node, Edge] =
    val laidOut = records.map(_.record(nodes, edges))
    def stacked[L: Label, V](read: FloorplanRecord[Node, Edge] => Tensor1[L, V]) = stack(laidOut.map(read), batch)
    FloorplanRecordBatch(
      stacked(_.nodes.nodeClass), stacked(_.nodes.startX), stacked(_.nodes.startY), stacked(_.nodes.endX), stacked(_.nodes.endY),
      stacked(_.edges.relationship), stacked(_.edges.subject), stacked(_.edges.obj)
    )

  /** The order that reads the positions by ascending `rank`, shuffled within a rank. */
  private def inOrder[L: Label](rank: Tensor1[L, Float32], key: Key): Tensor1[L, Int32] =
    val slots = rank.shape.extent(Axis[L])
    val shuffle = dimwit.Random.permutation(slots)(key).asFloat(VType[Float32])
    (rank *! slots.size.toFloat + shuffle).argsort(Axis[L])

/** Axis of the floor plans of a part. */
private trait Plans derives Label

/** DimWit wrapper around the synthetic floor plans
  * ([[https://huggingface.co/datasets/benikm91/synthetic-floorplans-for-transcription]]), backed by
  * ScalaPy. Its files are downloaded on first use and read from the Hugging Face cache after.
  *
  * Call `dimwit.initialize()` once before using this loader.
  *
  * {{{
  * dimwit.initialize()
  *
  * val data = SyntheticFloorplans.open(Axis[Width], Axis[Height], Axis[Channel], Axis[Node], Axis[Edge])(Split.Train, canvas = SyntheticFloorplans.storedSide)
  * val batch = data.batches(Axis[Batch] -> 64, afterSteps = 0).next()
  * }}}
  */
object SyntheticFloorplans:

  /** What a run names the corpus by on its command line and in what it writes. */
  val name = "synthetic-floorplans"
  val repoId = "jotamunz/synthetic-floorplans"

  /** The side the drawings are stored at in pixels, which a record's points are given in. */
  val storedSide = 512

  /** The most nodes and relationships a record of the training split holds, which every slot is
    * sized from.
    */
  val maxNodes = 73
  val maxRelationships = 101

  /** Opens a split, its drawings read at `canvas` pixels a side: [[storedSide]] or a divisor of it,
    * every pixel then the mean of the block it covers. A part is averaged down the first time it is
    * read at a side, and kept beside the Hugging Face cache. A record's points are fractions of the
    * side, so they are the same at any `canvas`.
    *
    * A split is read in its parts — train in nine of 50,000 plans — each read when it is reached, so
    * that only one part's records are held at a time; its drawings stay memory mapped.
    */
  def open[W: Label, H: Label, C: Label, Node: Label, Edge: Label](
      width: Axis[W],
      height: Axis[H],
      channel: Axis[C],
      node: Axis[Node],
      edge: Axis[Edge]
  )(split: Split, canvas: Int): SyntheticFloorplans[W, H, C, Node, Edge] =
    require(storedSide % canvas == 0, s"drawings of $storedSide pixels do not average down to $canvas")
    val parts = module.parts(repoId, split.fileName).as[Seq[py.Dynamic]].map: part =>
      (part.applyDynamic("__getitem__")(0).as[String], part.applyDynamic("__getitem__")(1).as[Int])
    require(parts.nonEmpty, s"no part of the split '${split.fileName}' in $repoId")
    new SyntheticFloorplans(parts, canvas)

  private[syntheticFloorplansDataset] lazy val module: py.Dynamic =
    PythonModules("synthetic_floorplans", "./syntheticFloorplansDataset/src/main/resources/python")

/** A single split of the synthetic floor plans. Use [[SyntheticFloorplans.open]] to create one. */
final class SyntheticFloorplans[W: Label, H: Label, C: Label, Node: Label, Edge: Label] private[syntheticFloorplansDataset] (
    /** Every part of the split, by name, and how many plans it holds, in seed order. */
    val parts: Seq[(String, Int)],
    /** The side the drawings are read at, in pixels. */
    val canvas: Int
):
  import SyntheticFloorplans.*

  val numSamples: Int = parts.map(_._2).sum

  /** Every floor plan of the split, once, in seed order. */
  def samples: Iterator[Sample[W, H, C, FloorplanRecord[Node, Edge]]] =
    parts.indices.iterator.flatMap: part =>
      val (images, parsed) = read(part)
      val records = recordsIn(parsed, Axis[Plans] -> parts(part)._2, 0)
      (0 until parts(part)._2).iterator.map: at =>
        val plan = Axis[Plans].at(at)
        val image = drawingsOf(liftPyTensor[(Plans, H, W), UInt8](Jax.jnp.asarray(images.applyDynamic("__getitem__")(py.Dynamic.global.slice(at, at + 1)))), Axis[C])
        Sample(
          image.squeeze(Axis[Plans]),
          FloorplanRecord(
            FloorplanNodes(records.nodeClass.slice(plan), records.startX.slice(plan), records.startY.slice(plan), records.endX.slice(plan), records.endY.slice(plan)),
            FloorplanEdges(records.relationship.slice(plan), records.subject.slice(plan), records.obj.slice(plan))
          )
        )

  /** Batches of floor plans, for as long as they are asked for, on the host: the parts in turn, each
    * cut into whole batches. The plans were generated independently of one another, so reading them
    * in order is already a shuffle.
    *
    * @param afterSteps The batches already taken, which are passed over without being read: a run
    *                   that carries on from a checkpoint reads on where it left off, and the parts
    *                   it passes are not read at all.
    */
  def batches[S: Label](batch: AxisExtent[S], afterSteps: Int): Iterator[Batch[S, W, H, FloorplanRecordBatch[S, Node, Edge]]] =
    val perPart = parts.map(_._2 / batch.size)
    require(perPart.forall(_ > 0), s"a batch of ${batch.size} exceeds a part of ${parts.map(_._2).min} plans")
    val skipped = afterSteps % perPart.sum
    val firstPart = perPart.scanLeft(0)(_ + _).init.lastIndexWhere(_ <= skipped)
    val intoFirst = skipped - perPart.take(firstPart).sum
    Iterator.from(firstPart).flatMap: at =>
      batchesOf(at % parts.size, batch, skipping = if at == firstPart then intoFirst else 0)

  /** The plans of one part in whole batches, in seed order, once: a part that does not cut into
    * whole batches leaves its last plans out.
    *
    * @param skipping The batches passed over without being read.
    */
  def batchesOf[S: Label](part: Int, batch: AxisExtent[S], skipping: Int = 0): Iterator[Batch[S, W, H, FloorplanRecordBatch[S, Node, Edge]]] =
    val (images, parsed) = read(part)
    val starts = (0 until parts(part)._2 / batch.size).map(_ * batch.size).drop(skipping)
    starts.iterator.map: from =>
      val pixels = images.applyDynamic("__getitem__")(py.Dynamic.global.slice(from, from + batch.size))
      Batch(liftPyTensor[(S, H, W), UInt8](pixels), recordsIn(parsed, batch, from))

  /** The seed of every plan of a part, in seed order: what names a plan in every dataset made from
    * the same generator.
    */
  def seeds(part: Int): Seq[Int] = module.seeds(repoId, parts(part)._1).as[Seq[Int]]

  override def toString: String = s"SyntheticFloorplans($repoId, plans=$numSamples, parts=${parts.size}, canvas=$canvas)"

  /** A part's drawings, memory mapped, and its records, parsed on the host: each read the first time
    * it is asked for and kept. Training comes back to every part many times, and parsing a part's
    * records takes far longer than training on them.
    */
  private final class Part(name: String):
    lazy val drawings: py.Dynamic = module.drawings(repoId, name, canvas)
    lazy val records: py.Dynamic = module.records(repoId, name, storedSide, maxNodes, maxRelationships,
      FloorplanClass.values.toSeq.map(_.recordName).toPythonCopy, FloorplanRelationship.values.toSeq.map(_.recordName).toPythonCopy)

  private val held: IndexedSeq[Part] = parts.map((name, _) => Part(name)).toIndexedSeq

  private def read(part: Int): (py.Dynamic, py.Dynamic) = (held(part).drawings, held(part).records)

  /** The records of the plans from `from` on, cut from the parsed ones. */
  private def recordsIn[S: Label](parsed: py.Dynamic, rows: AxisExtent[S], from: Int): FloorplanRecordBatch[S, Node, Edge] =
    val taken = py.Dynamic.global.slice(from, from + rows.size)
    def cut(at: Int) = parsed.applyDynamic("__getitem__")(at).applyDynamic("__getitem__")(taken)
    FloorplanRecordBatch(
      nodeClass = liftPyTensor[(S, Node), Int32](cut(0)),
      startX = liftPyTensor[(S, Node), Float32](cut(1)),
      startY = liftPyTensor[(S, Node), Float32](cut(2)),
      endX = liftPyTensor[(S, Node), Float32](cut(3)),
      endY = liftPyTensor[(S, Node), Float32](cut(4)),
      relationship = liftPyTensor[(S, Edge), Int32](cut(5)),
      subject = liftPyTensor[(S, Edge), Int32](cut(6)),
      obj = liftPyTensor[(S, Edge), Int32](cut(7))
    )
