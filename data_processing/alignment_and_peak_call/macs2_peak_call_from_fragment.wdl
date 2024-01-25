version 1.0

workflow peak_calling {
  input {
    File in_fragments
    File peaks_merge_script
    Int half_width
    Float norm_factor
    File blacklist
    Int pos_shift
    Int neg_shift
    String sample_id
    
    Float q_value_cutoff = if (norm_factor >= 0.0 && norm_factor < 5.0) then 0.01
                          else if (norm_factor >= 5.0 && norm_factor < 25.0) then 0.005
                          else if (norm_factor >= 25.0 && norm_factor < 50.0) then 0.0025
                          else if (norm_factor >= 50.0 && norm_factor < 100.0) then 0.001
                          else 0.0005
    
    Int macs2_disk_space
    Int macs2_num_threads
    Int macs2_num_preempt
    Int macs2_memory
    
    Int peaks_center_merge_disk_space
    Int peaks_center_merge_num_threads
    Int peaks_center_merge_num_preempt
    Int peaks_center_merge_memory
  }
  
  call macs2 {
    input:
      in_fragments = in_fragments,
      norm_factor=norm_factor,
      q_value_cutoff=q_value_cutoff,
      pos_shift=pos_shift,
      neg_shift=neg_shift,
      sample_id=sample_id,
      disk_space=macs2_disk_space,
      num_threads=macs2_num_threads,
      num_preempt=macs2_num_preempt,
      memory=macs2_memory
  }
    
  call peaks_center_merge {
    input:
      peak_file=macs2.narrow_peaks,
      peak_file_pr1=macs2.narrow_peaks_pr1,
      peak_file_pr2=macs2.narrow_peaks_pr2,
      blacklist=blacklist,
      half_width=half_width,
      peaks_merge_script=peaks_merge_script,
      sample_id=sample_id,
      disk_space=peaks_center_merge_disk_space,
      num_threads=peaks_center_merge_num_threads,
      num_preempt=peaks_center_merge_num_preempt,
      memory=peaks_center_merge_memory
  }

  output {
    File peaks = peaks_center_merge.merged_peaks
    Int num_peaks = peaks_center_merge.num_peaks
    Float q_value_output = macs2.q_value_output
  }

}


task macs2 {
  input {
    File in_fragments
    Float norm_factor
    String sample_id
    Int pos_shift
    Int neg_shift
    Int disk_space
    Int num_threads
    Int num_preempt
    Int memory
    
    String lbrace = "{"
    String rbrace = "}"
  
    Float q_value_cutoff
  }

  command {  
    set -euo pipefail
    zcat ${in_fragments} | awk '${lbrace}OFS="\t"${rbrace}${lbrace} print $1,$2+${pos_shift},$2+${pos_shift}+1 ${rbrace}' | gzip > fwd.bed.gz
    zcat ${in_fragments} | awk '${lbrace}OFS="\t"${rbrace}${lbrace} print $1,$3-${neg_shift},$3-${neg_shift}+1 ${rbrace}' | gzip > rev.bed.gz
    zcat fwd.bed.gz rev.bed.gz | sort -k1,1 -k2,2n | shuf > ${sample_id}.bed
    
    total_lines=$(wc -l < "${sample_id}.bed")
    split_point=$((total_lines / 2))
    
    head -n "$split_point" ${sample_id}.bed > subset1.bed
    tail -n +"$((split_point + 1))" ${sample_id}.bed > subset2.bed
    
    macs2 callpeak -t subset1.bed -f BED -g hs --shift -75 --extsize 150 --nomodel --call-summits --nolambda --keep-dup all -q ${q_value_cutoff} -n ${sample_id}_subset1
    macs2 callpeak -t subset2.bed -f BED -g hs --shift -75 --extsize 150 --nomodel --call-summits --nolambda --keep-dup all -q ${q_value_cutoff} -n ${sample_id}_subset2
    macs2 callpeak -t ${sample_id}.bed -f BED -g hs --shift -75 --extsize 150 --nomodel --call-summits --nolambda --keep-dup all -q ${q_value_cutoff} -n ${sample_id}
  }

  runtime {
    docker: "fooliu/macs2"
    memory: "${memory}GB"
    cpu: "${num_threads}"
    disks: "local-disk ${disk_space} HDD"
    preemptible: "${num_preempt}"
  }

  output {
    File narrow_peaks = "${sample_id}_peaks.narrowPeak"
    File narrow_peaks_pr1 = "${sample_id}_subset1_peaks.narrowPeak"
    File narrow_peaks_pr2 = "${sample_id}_subset2_peaks.narrowPeak"
    Float q_value_output = q_value_cutoff
  }
}

task peaks_center_merge {
  input {
    File peak_file
    File peak_file_pr1
    File peak_file_pr2
    File blacklist
    String sample_id
    File peaks_merge_script
	Int half_width
    Int disk_space
    Int num_threads
    Int num_preempt
    Int memory
    String lbrace = "{"
    String rbrace = "}"
  }
  
  command {
    set -euo pipefail
    
    cat ${peak_file} | grep 'chr' | grep -v 'chrKI\|chrGL\|chrEBV\|chrM\|chrMT\|_KI\|_GL'| bedtools intersect -a - -b ${blacklist} -v -wa | awk '${lbrace}OFS="\t"${rbrace}${lbrace}print $1,$2+$10-${half_width},$2+$10+${half_width},$4,$9,$10${rbrace}' | sort -k1,1 -k2,2n > peak_file_overall.bed

    cat ${peak_file_pr1} | grep 'chr' | grep -v 'chrKI\|chrGL\|chrEBV\|chrM\|chrMT\|_KI\|_GL' | bedtools intersect -a - -b ${blacklist} -v -wa | awk '${lbrace}OFS="\t"${rbrace}${lbrace}print $1,$2+$10-${half_width},$2+$10+${half_width},$4,$9,$10${rbrace}' | sort -k1,1 -k2,2n > peak_pr1.bed
    cat ${peak_file_pr2} | grep 'chr' | grep -v 'chrKI\|chrGL\|chrEBV\|chrM\|chrMT\|_KI\|_GL' | bedtools intersect -a - -b ${blacklist} -v -wa | awk '${lbrace}OFS="\t"${rbrace}${lbrace}print $1,$2+$10-${half_width},$2+$10+${half_width},$4,$9,$10${rbrace}' | sort -k1,1 -k2,2n > peak_pr2.bed
    

    bedtools intersect -a peak_file_overall.bed -b peak_pr1.bed -wa -f 0.50 -u > peak_intersect_pr1.bed
    bedtools intersect -a peak_intersect_pr1.bed -b peak_pr2.bed -wa -f 0.50 -u | gzip > ${sample_id}.pr.temp.peaks.bed.gz
   
    mv ${peaks_merge_script} peaks_merge_script.py
    python peaks_merge_script.py ${sample_id}.pr.temp.peaks.bed.gz
    gzip all_peaks_collapsed.bed
    mv all_peaks_collapsed.bed.gz ${sample_id}.pr.peaks.bed.gz
    
    zcat ${sample_id}.pr.peaks.bed.gz | wc -l > peak_count
    
    
  }

  output {
    File merged_peaks = "${sample_id}.pr.peaks.bed.gz"
    Int num_peaks = read_int("peak_count")
  }

  runtime {
    docker: "njaved/samtools_bedtools"
    memory: "${memory}GB"
    cpu: "${num_threads}"
    disks: "local-disk ${disk_space} HDD"
    preemptible: "${num_preempt}"
  }
}