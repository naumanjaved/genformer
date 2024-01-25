version 1.0

workflow bedtools_bamtobed {
  input {
    File in_bed
    String sample_id
    File chain_file
    File liftover_utility
    Int disk_space
    Int num_threads
    Int num_preempt
    Int memory
    
  }


  call liftover {input:
    in_bed=in_bed,
    num_threads=num_threads,
    sample_id=sample_id,
    chain_file=chain_file,
    liftover_utility=liftover_utility,
    disk_space=disk_space,
    num_preempt=num_preempt,
    memory=memory
  }

  output {
    File liftover_fragments = liftover.liftover_fragments
  }
}



task liftover {
  input {
    File in_bed
    String sample_id
    Int disk_space
    Int num_threads
    Int num_preempt
    Int memory
    File liftover_utility
    File chain_file
    
    String lbrace = "{"
    String rbrace = "}"
    
  }

  command {
    zcat ${in_bed} > unzipped.bed
    mv ${liftover_utility} liftOver
    chmod a+x liftOver
    ./liftOver unzipped.bed ${chain_file} ${sample_id}.frag.bed unmapped.bed
    gzip ${sample_id}.frag.bed
  }

  output {
    File liftover_fragments = "${sample_id}.frag.bed.gz"
  }

  runtime {
    docker : "njaved/tensorflow2.6_records_env:latest"
    memory: "${memory}GB"
    cpu: "${num_threads}"
    disks: "local-disk ${disk_space} HDD"
    preemptible: "${num_preempt}"
  }
}