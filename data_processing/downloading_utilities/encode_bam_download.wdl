version 1.0

workflow encode_bam_download {

  input {
    String accessions
    String sample_id
    String assay

    Float memory = 2
    Int disk_space = 200
    Int num_threads = 1
    Int num_preempt = 2

  }
  
  call strsplit {
    input:accessions=accessions
  }
  
  scatter(i in range(length(strsplit.bams))) {
    call download { input :
        accession=strsplit.bams[i],
        num_threads=num_threads,
        disk_space=disk_space,
        num_preempt=num_preempt,
        memory=memory
    }
  }

  call merge { input :
      in_bams=download.bams,
      sample_id=sample_id,
      assay=assay,
      num_threads=num_threads,
      disk_space=disk_space,
      num_preempt=num_preempt,
      memory=memory
  }


  output {
    Array[File] bams=merge.bams
  }
}

task strsplit {
  input {
    String accessions
    Int disk_space = 5
    Int num_threads = 1
    Int num_preempt = 1
    Float memory = 1
  }

  command {
    echo ${accessions} | sed 's(,(\n(g' >> bams.txt
  }


  output {
    Array[String] bams = read_lines("bams.txt")
  }

  runtime {
    docker: "njaved/minimap2"
    memory: "${memory}GB"
    cpu: "${num_threads}"
    disks: "local-disk ${disk_space} HDD"
    preemptible: "${num_preempt}"
  }
}

task download {
  input {
    String accession
    Int disk_space
    Int num_threads
    Int num_preempt
    Float memory
  }

  command {
    set -euo pipefail

    wget https://www.encodeproject.org/files/${accession}/@@download/${accession}.bam
    samtools sort -@ ${num_threads} ${accession}.bam -o ${accession}.sort.bam 
  }

  output {
    File bams = "${accession}.sort.bam"

  }

  runtime {
    docker: "njaved/samtools_bedtools"
    memory: "${memory}GB"
    cpu: "${num_threads}"
    disks: "local-disk ${disk_space} HDD"
    preemptible: "${num_preempt}"
  }
}


task merge {
  input {
    Array[File] in_bams
    File first_bam = select_first(in_bams)
    Int num_bams = length(in_bams)
	String sample_id
    String assay
    Int disk_space
    Int num_threads
    Int num_preempt
    Float memory
  }
  command {
    if (("${num_bams}" > 1))
    then
      samtools merge -@ ${num_threads} ${sample_id}.${assay}.merged.bam ${sep=' ' in_bams}
    else
      mv ${first_bam} ${sample_id}.${assay}.merged.bam
    fi
  }

  output {
    Array[File] bams = glob("*.merged.bam")
  }

  runtime {
    docker : "njaved/deeptools"
    memory: "${memory}GB"
    cpu: "${num_threads}"
    disks: "local-disk ${disk_space} HDD"
    preemptible: "${num_preempt}"
  }
}