package d2g.floorplan.eval

import d2g.*
import d2g.floorplan.config.D2GSetup
import d2g.floorplan.train.SavedParams
import dataset.DrawingDataset.Split
import dataset.Runs
import dataset.drawingsOf
import deepwit.checkpointing.TensorTreeCheckpointer
import dimwit.*
import dimwit.jax.Jax
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.SeqConverters
import syntheticFloorplansDataset.FloorplanGraph
import syntheticFloorplansDataset.SyntheticFloorplans

import java.nio.file.Files
import java.nio.file.Path

import scala.language.implicitConversions

/** Writes what the last checkpoint of the newest run makes of every plan, a folder per part: the
  * encoders patch embeddings, the predicted records, and the prediction embedding each of their nodes
  * and relationships was written from. A part already written is passed over.
  */
def embeddingsTranscriber(setup: D2GSetup, size: String): Unit =
  dimwit.initialize()

  val checkpoints = TensorTreeCheckpointer.latestIn(setup.checkpointRoot).getOrElse(sys.error(s"no training run in ${setup.checkpointRoot}"))
  println(s"reading ${checkpoints.rootPath}")
  val params = checkpoints.loadLatest[SavedParams].getOrElse(sys.error(s"no checkpoint in ${checkpoints.rootPath}")).params
  val transcriber = Transcriber(Axis[Node] -> setup.nodeSlots, Axis[Edge] -> setup.edgeSlots)
  val directory = Path.of(Runs.outputDir, s"d2g-embeddings-${SyntheticFloorplans.name}-$size")
  println(s"writing to $directory")

  def numpy[T <: Tuple: Labels, V](tensor: Tensor[T, V]) = TensorTree[Tensor[T, V]].toNumpyTree(tensor)

  /** A batch as it is kept until its part is saved */
  case class Embedded(
      records: Seq[String],
      patches: Tensor3[Drawing, Patch, Embedding, Float16],
      nodes: Tensor2[Node, Embedding, Float16],
      nodeCounts: Seq[Int],
      relationships: Tensor2[Edge, Embedding, Float16],
      relationshipCounts: Seq[Int]
  )

  Seq(Split.Validation, Split.Test, Split.Train).foreach: split =>
    val plans = open(setup, split)
    plans.parts.indices.filterNot(part => Files.isDirectory(directory.resolve(plans.parts(part)._1))).foreach: part =>
      val (name, planCount) = plans.parts(part)
      require(planCount % EmbeddedTogether == 0, s"$name holds $planCount plans, which do not cut into batches of $EmbeddedTogether")
      val seeds = plans.seeds(part)
      val embedded = plans.batchesOf(part, Axis[Drawing] -> EmbeddedTogether).zip(seeds.grouped(EmbeddedTogether)).map: (batch, batchSeeds) =>
        val transcribed = transcriber.transcription(params, drawingsOf(batch.pixels, Axis[Channel]))
        val records = FloorplanGraph.of(transcribed.record)
        val (nodeCounts, relationshipCounts) = (records.map(_.nodes.size), records.map(_.edges.size))
        val (nodePredictions, relationshipPredictions) = (transcribed.nodePredictions.asFloat(VType[Float16]), transcribed.relationshipPredictions.asFloat(VType[Float16]))
        Embedded(
          records.zip(batchSeeds).map((record, seed) => s"""{"seed": $seed, ${record.recordFields}}"""),
          transcribed.patches.asFloat(VType[Float16]),
          concatenate(nodeCounts.zipWithIndex.map((count, plan) => nodePredictions.slice(Axis[Drawing].at(plan)).slice(Axis[Node].at(0 until count))), Axis[Node]),
          nodeCounts,
          concatenate(relationshipCounts.zipWithIndex.map((count, plan) => relationshipPredictions.slice(Axis[Drawing].at(plan)).slice(Axis[Edge].at(0 until count))), Axis[Edge]),
          relationshipCounts
        )
      .toSeq

      /* Row `i` of a part is about the plan `seeds[i]`, the seed naming it in the dataset:
       *   records.jsonl: the record written down, as the dataset's labels are
       *   patches.npy: (plans, patches, embedding) the encoded document
       *   node_predictions.npy: (nodes, embedding) one per node of every record, in its order;
       *                         plan i's are rows node_offsets[i] until node_offsets[i + 1]
       *   relationship_predictions.npy: the same for the relationships, with relationship_offsets.npy
       *
       * Written under a name of its own and moved once whole, so that a part under its own name is
       * always complete.
       */
      val writing = Files.createDirectories(directory.resolve(s"$name.partial"))
      def save(file: String, array: py.Any) = Jax.np.save(writing.resolve(file).toString, array)
      def offsets(counts: Seq[Int]) = Jax.np.array(counts.scanLeft(0)(_ + _).toPythonCopy, dtype = "int64")
      Files.writeString(writing.resolve("records.jsonl"), embedded.flatMap(_.records).mkString("", "\n", "\n"))
      save("seeds.npy", Jax.np.array(seeds.toPythonCopy, dtype = "int64"))
      save("patches.npy", numpy(concatenate(embedded.map(_.patches), Axis[Drawing])))
      save("node_predictions.npy", numpy(concatenate(embedded.map(_.nodes), Axis[Node])))
      save("node_offsets.npy", offsets(embedded.flatMap(_.nodeCounts)))
      save("relationship_predictions.npy", numpy(concatenate(embedded.map(_.relationships), Axis[Edge])))
      save("relationship_offsets.npy", offsets(embedded.flatMap(_.relationshipCounts)))
      val written = Files.move(writing, directory.resolve(name))
      println(s"$name | $planCount plans written to $written")

/** How many plans are transcribed together: one traced computation. */
private val EmbeddedTogether = 500
