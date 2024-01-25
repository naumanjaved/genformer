version 1.0

workflow bedtools_bamtobed {
  input {
    String in_bam
    String sample_id
    Float paired_scale_factor
    File genome_file
    Int disk_space
    Int num_threads
    Int num_preempt
    Int memory
    Int disk_space_frag
    Int num_threads_frag
    Int num_preempt_frag
    Int memory_frag

  }

  call bamtobed {input:
    in_bam=in_bam,
    paired_scale_factor=paired_scale_factor,
    num_threads=num_threads,
    sample_id=sample_id,
    disk_space=disk_space,
    num_preempt=num_preempt,
    memory=memory
  }
  
  call bedtools {input:
    in_bam=in_bam,
    sample_id=sample_id,
    scale_factor=bamtobed.num_fragments,
    num_threads=num_threads_frag,
    sample_id=sample_id,
    genome_file=genome_file,
    disk_space=disk_space_frag,
    num_preempt=num_preempt_frag,
    memory=memory_frag
  }

  output {
    File bedgraph = bedtools.bedgraph
    Float adj_scale_factor = bedtools.adj_scale_factor}
}

task bamtobed {
  input {
    String in_bam
    String sample_id
    Float paired_scale_factor
    Int disk_space
    Int num_threads
    Int num_preempt
    Int memory
    String lbrace = "{"
    String rbrace = "}"
    
  }

  command {
    wget -O downloaded.bam ${in_bam}
    bedtools bamtobed -i /cromwell_root/downloaded.bam | awk '$5 > 0' | gzip > ${sample_id}.bed.gz
    
    zcat ${sample_id}.bed.gz | wc -l | awk '${lbrace} print $1 / (${paired_scale_factor}*1000000.0) ${rbrace}' > test.num_fragments.out
    
  }

  output {
    File fragments = "${sample_id}.bed.gz"
    Float num_fragments = read_float("test.num_fragments.out")
  }

  runtime {
    docker : "njaved/samtools_bedtools"
    memory: "${memory}GB"
    cpu: "${num_threads}"
    disks: "local-disk ${disk_space} HDD"
    preemptible: "${num_preempt}"
  }
}

task bedtools {
  input {
    String in_bam
    String sample_id
    Float scale_factor
    Float scale_factor_inv = 1.0 / (scale_factor)
	File genome_file
    Int disk_space
    Int num_threads
    Int num_preempt
    Int memory
    String lbrace = "{"
    String rbrace = "}"
  }
  command {
  
    wget -O downloaded.bam ${in_bam}
    
    samtools view -bf 0x2 downloaded.bam | samtools sort -n | bedtools bamtobed -i stdin -bedpe -mate1 > temp.bedpe
    

    awk 'BEGIN ${lbrace}OFS="\t"${rbrace} ${lbrace}if ($9 == "+") ${lbrace}print $1, $2-5, $2+5 ${rbrace} else if ($9 == "-") ${lbrace}print $1, $3-5, $3+5${rbrace} ${rbrace}' temp.bedpe | grep -v 'KI\|GL\|EBV\|chrM\|chrMT\|K\|J\|phi\|ERCC'| sort -k1,1 -k2,2n > out.bed
    bedtools genomecov -i out.bed -g ${genome_file} -bg -scale ${scale_factor_inv} | sort -k1,1 -k2,2n | gzip > ${sample_id}.bedgraph.gz
  }

  output {
    File bedgraph = "${sample_id}.bedgraph.gz"
    Float adj_scale_factor = "${scale_factor}"
  }

  runtime {
    docker : "njaved/samtools_bedtools"
    memory: "${memory}GB"
    cpu: "${num_threads}"
    disks: "local-disk ${disk_space} HDD"
    preemptible: "${num_preempt}"
  }
}