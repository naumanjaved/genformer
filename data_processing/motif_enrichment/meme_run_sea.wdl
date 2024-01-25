version 1.0

workflow meme_run_sea {
  input {
    File motif_file
    File peak_file
    File background_peaks
    File genome_fasta
    Float thresh
    String input_name
    Int top_peaks
    Int half_peak_width

    Int bed_disk_space
    Int bed_num_threads
    Int bed_num_preempt
    Int bed_memory

    Int meme_disk_space
    Int meme_num_threads
    Int meme_num_preempt
    Int meme_memory

  }

  call bed_getfasta {input:
    peak_file=peak_file,
    top_peaks=top_peaks,
    half_peak_width=half_peak_width,
    background_peaks=background_peaks,
    genome_fasta=genome_fasta,
    num_threads=bed_num_threads,
    disk_space=bed_disk_space,
    num_preempt=bed_num_preempt,
    memory=bed_memory
  }

  call meme {input:
    motif_file=motif_file,
    input_name=input_name,
    bg_sequences=bed_getfasta.bg_peaks,
    peak_sequences=bed_getfasta.outfasta,
    thresh=thresh,
    num_threads=meme_num_threads,
    disk_space=meme_disk_space,
    num_preempt=meme_num_preempt,
    memory=meme_memory
  }

  output {
    File out_tsv = meme.out_tsv
  }
}

task bed_getfasta {
  input {
    File peak_file
    File background_peaks
    File genome_fasta

	Int top_peaks
    Int half_peak_width
    Int disk_space
    Int num_threads
    Int num_preempt
    Int memory
    
    String lbrace = "{"
    String rbrace = "}"
  }
  command {
    zcat ${peak_file} | sort -k5,5nr | head -${top_peaks} | awk '${lbrace} OFS = "\t" ${rbrace}${lbrace} print $1,$2+${half_peak_width},$3-${half_peak_width}${rbrace}' | sort -k1,1 -k2,2n > sorted.peaks.bed
    bedtools getfasta -fi ${genome_fasta} -bed sorted.peaks.bed > merged.fasta
    bedtools getfasta -fi ${genome_fasta} -bed ${background_peaks} > bg_peaks.fasta
  }

  output {
    File outfasta = "merged.fasta"
    File bg_peaks = "bg_peaks.fasta"
  }

  runtime {
    docker : "njaved/samtools_bedtools"
    memory: "${memory}GB"
    cpu: "${num_threads}"
    disks: "local-disk ${disk_space} HDD"
    preemptible: "${num_preempt}"
  }
}

task meme {
  input {
    File motif_file
    String input_name
    File peak_sequences
    File bg_sequences
    Float thresh

    Int disk_space
    Int num_threads
    Int num_preempt
    Int memory
  }
  command {
    sea --p ${peak_sequences} --m ${motif_file} --n ${bg_sequences} --thresh ${thresh} --verbosity 1
    mv sea_out/sea.tsv sea_out/${input_name}.tsv
  }

  output {
    File out_tsv = "sea_out/${input_name}.tsv"
  }

  runtime {
    docker : "memesuite/memesuite:5.4.1"
    memory: "${memory}GB"
    cpu: "${num_threads}"
    disks: "local-disk ${disk_space} HDD"
    preemptible: "${num_preempt}"
  }
}