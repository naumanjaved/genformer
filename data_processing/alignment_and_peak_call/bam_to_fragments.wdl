version 1.0

workflow bedtools_bamtobed {
  input {
    File in_bam
    String sample_id
    Int disk_space
    Int num_threads
    Int num_preempt
    Int memory
    
  }


  call bamtobed {input:
    in_bam=in_bam,
    num_threads=num_threads,
    sample_id=sample_id,
    disk_space=disk_space,
    num_preempt=num_preempt,
    memory=memory
  }

  output {
    File fragments = bamtobed.fragments
    Float fpkm_scaling_factor = bamtobed.num_fragments}
}



task bamtobed {
  input {
    File in_bam
    String sample_id
    Int disk_space
    Int num_threads
    Int num_preempt
    Int memory
    
    String lbrace = "{"
    String rbrace = "}"
    
  }

  command {
    samtools sort -n ${in_bam} -@ 1 -m 2G -o /cromwell_root/named_sorted.bam

    bedtools bamtobed -i /cromwell_root/named_sorted.bam -bedpe | awk '$1 == $4' | awk '$8 >= 20' | awk -v OFS="\t" '${lbrace}if($9 == "+")${lbrace}print $1,$2+4,$6-5${rbrace}else if($9=="-")${lbrace}print $1,$5+4,$3-5${rbrace}${rbrace}' | awk -v OFS="\t" '${lbrace}if($3<$2)${lbrace}print $1,$3,$2${rbrace}else if($3>$2)${lbrace}print $1,$2,$3${rbrace}${rbrace}' | awk '$3-$2 > 0' | gzip > ${sample_id}.bed.gz
    zcat ${sample_id}.bed.gz | wc -l | awk '${lbrace} print $1 / 1000000.0 ${rbrace}' > test.num_fragments.out
    
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