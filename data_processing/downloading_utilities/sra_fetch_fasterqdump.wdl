version 1.0

workflow fetch_sra {

  input {
    String SRR_IDs

    Int disk_space
    Int num_threads
    Int num_preempt
    Float memory
  }
  call strsplit {
    input:SRR_IDs=SRR_IDs
  }
  scatter(SRR_ID in strsplit.SRR_IDs_arr) {
    call prefetch_fastq_dump {
      input:SRR_ID=SRR_ID, disk_space=disk_space, num_threads=num_threads,
      num_preempt=num_preempt, num_threads=num_threads,memory=memory
    }
  }


  output {
    Array[File] read1=prefetch_fastq_dump.read1
    Array[File?] read2=prefetch_fastq_dump.read2
  }
}

task strsplit {
  input {
    String SRR_IDs

    Int disk_space = 5
    Int num_threads = 1
    Int num_preempt = 1
    Float memory = 1
  }

  command {
    echo ${SRR_IDs} | sed 's(,(\n(g' >> IDs.txt

  }
  output {
    Array[String] SRR_IDs_arr = read_lines("IDs.txt")
  }

  runtime {
    docker: "njaved/fastq_download:latest"
    memory: "${memory}GB"
    cpu: "${num_threads}"
    disks: "local-disk ${disk_space} HDD"
    preemptible: "${num_preempt}"
  }
}

task prefetch_fastq_dump {
  input {
    String SRR_ID
    
    Int disk_space
    Int num_threads
    Int num_preempt
    Float memory
  }

  command {
    ls -lahtr
    prefetch ${SRR_ID}
    fasterq-dump --split-3 ${SRR_ID} --outdir . --temp . --skip-technical 
    pigz *.fastq
    if !(test -f ${SRR_ID}_1.fastq.gz); then
        mv ${SRR_ID}.fastq.gz ${SRR_ID}_1.fastq.gz
    fi
    
  }
  output {
    File read1="${SRR_ID}_1.fastq.gz"
    File? read2="${SRR_ID}_2.fastq.gz"
  }

  runtime {
    docker: "njaved/sra-tools-ubuntu:latest"
    memory: "${memory}GB"
    cpu: "${num_threads}"
    disks: "local-disk ${disk_space} HDD"
    preemptible: "${num_preempt}"
  }
}