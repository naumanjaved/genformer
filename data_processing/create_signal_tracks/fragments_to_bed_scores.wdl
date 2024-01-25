version 1.0

workflow bedtools_bamtobed {
  input {
    File fragments
    String sample_id
    Float scale_factor
    File genome_file
    Int disk_space
    Int num_threads
    Int num_preempt
    Int memory
    Int pos_shift
    Int neg_shift
    
  }


  call bamtobed {input:
    fragments=fragments,
    scale_factor=scale_factor,
    num_threads=num_threads,
    genome_file=genome_file,
    pos_shift=pos_shift,
    neg_shift=neg_shift,
    sample_id=sample_id,
    disk_space=disk_space,
    num_preempt=num_preempt,
    memory=memory
  }

  output {
    File tn5_bedgraph = bamtobed.tn5_bedgraph}
}



task bamtobed {
  input {
    File fragments
    File genome_file
    Float scale_factor
    Float scale_factor_inv = 20.0 / scale_factor
    String sample_id
    Int disk_space
    Int num_threads
    Int num_preempt
    Int memory
    
    String lbrace = "{"
    String rbrace = "}"
    
    Int pos_shift
    Int neg_shift
    
  }

  command {
    
    zcat ${fragments} | awk '${lbrace}OFS="\t"${rbrace}${lbrace} print $1,$2+${pos_shift},$2+${pos_shift}+1 ${rbrace}' | gzip > fwd.bed.gz
    zcat ${fragments} | awk '${lbrace}OFS="\t"${rbrace}${lbrace} print $1,$3-${neg_shift},$3-${neg_shift}+1 ${rbrace}' |gzip > rev.bed.gz
    zcat fwd.bed.gz rev.bed.gz | sort -k1,1 -k2,2n > ${sample_id}.bed

    cat ${sample_id}.bed | awk '${lbrace} OFS="\t" ${rbrace} ${lbrace} print $1,$2-5,$3+5 ${rbrace}' | grep -v 'KI\|GL\|EBV\|chrM\|chrMT\|K\|J' | awk '$2 >= 0' | sort -k1,1 -k2,2n | bedtools genomecov -i - -g ${genome_file} -scale ${scale_factor_inv} -bg | sort -k1,1 -k2,2n | gzip > ${sample_id}.bedgraph.gz
    
  }

  output {
    File tn5_bedgraph = "${sample_id}.bedgraph.gz"
  }

  runtime {
    docker : "njaved/samtools_bedtools"
    memory: "${memory}GB"
    cpu: "${num_threads}"
    disks: "local-disk ${disk_space} HDD"
    preemptible: "${num_preempt}"
  }
}