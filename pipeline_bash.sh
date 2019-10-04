#!/bin/bash

#mounting data volumes (if automount not set up yet)

sudo mount /dev/xvdi /mnt/disks/data/data1
sudo mount /dev/xvdf /mnt/disks/data/data2
sudo mount /dev/xvdg /mnt/disks/data/data3
sudo mount /dev/xvdj /mnt/disks/data/data4
sudo mount /dev/xvdk /mnt/disks/data/data5
sudo mount /dev/xvdh /mnt/disks/data/data6

#setting up ram for the processes

sudo blockdev --setra 16384 /dev/xvdf
sudo blockdev --setra 16384 /dev/xvdg
sudo blockdev --setra 16384 /dev/xvdi
sudo blockdev --setra 16384 /dev/xvdh
sudo blockdev --setra 16384 /dev/xvdk
sudo blockdev --setra 16384 /dev/xvdj

#prepare the disk resources and GPUs
sudo /usr/local/GPUtools/prepare_script.sh

#running BWA-MEM alignment with Picard markdups and BQSR on a tumor sample

sudo nohup pbrun fq2bam --ref /data/Ref/hg19_ref_genome.fa --in-fq /data/XXXXXXX_2_XXXXX_FXXXXXX_Homo-sapiens__R_TTTTT_XXXX_LXXXXXX_R1.fastq.gz /data/XXXXXXX_2_XXXXXX_XXXXXXX_Homo-sapiens__R_TTTTT_XXXXXXX_XXXXXXX_M025_R2.fastq.gz --in-fq /data/XXXXXXX_3_180115_XXXXXXX_Homo-sapiens__R_TTTTT_XXXXXXX_XXXXXXX_M025_R1.fastq.gz /data/XXXXXXX_3_TTTTT_XXXXXXX_Homo-sapiens__R_TTTTT_XXXXXXX_XXXXXXX_M025_R2.fastq.gz --in-fq /data2/XXXXXXX_4_180115_XXXXXXX_Homo-sapiens__R_TTTTT_XXXXXXX_XXXXXXX_M025_R1.fastq.gz /data2/XXXXXXX_4_TTTTT_XXXXXXX_Homo-sapiens__R_TTTTT_XXXXXXX_XXXXXXX_M025_R2.fastq.gz --out-bam /data/bam_output/XXXXXXX_180115_XXXXXXX_Homo-sapiens_tumor_sample1.bam &>/data/logs/sample1_fq2bam_tumor.log &
sample1_fq2bam_tumor_pid=$!

wait ${sample1_fq2bam_tumor_pid}

#clear disk  cache

sync
echo 1 | sudo tee /proc/sys/vm/drop_caches
echo 2 | sudo tee /proc/sys/vm/drop_caches
echo 3 | sudo tee /proc/sys/vm/drop_caches

#running BWA-MEM alignment with Picard Markdups and BQSR on a normal sample
sudo nohup pbrun fq2bam --ref /data/Ref/hg19_ref_genome.fa --in-fq /data/XXXXXXX_2_XXXXX_FXXXXXX_Homo-sapiens_XXXX_LXXXXXX_R1.fastq.gz /data/XXXXXXX_XXXXXX_XXXXXXX_Homo-sapiens__XXXXXXX_XXXXXXX_M025_R2.fastq.gz --out-bam /data/bam_output/XXXXXXX_XXXXXXX_Homo-sapiens_normal_sample1.bam --tmp-dir /data6/tmp/ &>/data/logs/sample1_fq2bam_normal.log &
sample1_fq2bam_normal_pid=$!


wait ${sample1_fq2bam_normal_pid}

#clear disk  cache

sync
echo 1 | sudo tee /proc/sys/vm/drop_caches
echo 2 | sudo tee /proc/sys/vm/drop_caches
echo 3 | sudo tee /proc/sys/vm/drop_caches


#run mutect2 on tumor normal

pbrun mutectcaller --ref /data/Ref/hg19_ref_genome.fa \
--in-tumor-bam /mnt/disks/big_disk/bam_output/XXXXXX_TTTTTT_XXXXXXXX_M042_tumor_sample1.bam \
--tumor-name sample1t \
--in-normal-bam /mnt/disks/bam_disk/output/XXXXXXX_TTTTTT_XXXXXXX_Homo-sapiens_normal_sample1.bam \
--normal-name sample1n \
--out-vcf /mnt/disks/big_disk/bam_output/XXXXXXXX_1XTTTTT_XXXXXX_M042_sample1_tumor_normal.vcf --tmp-dir /data6/tmp/ &>/data/logs/sample1_tn_mutectcaller_log.log &


#running deepvariant on tumor sample
pbrun deepvariant --ref /mnt/disks/data/big_disk/Ref/hg19_ref_genome.fa \
  --in-bam /data/XXXXXXX_1TTTTT_XXXXXXX_Homo-sapiens_tumor_sample1.bam \
  --out-variants /mnt/disks/data/data_output/vcfs/deepvariant_sample1_t.vcf


#running deepvariant on normal sample
pbrun deepvariant --ref /mnt/disks/data/big_disk/Ref/hg19_ref_genome.fa \
  --in-bam /data/XXXXXXX_1TTTTTT_XXXXXXX_Homo-sapiens_normal_sample1.bam \
  --out-variants /mnt/disks/data/data_output/vcfs/deepvariant_sample1_n.vcf


#postprocessing of deepvariant calls to determine somatic status.


deep_analysis /mnt/disks/data/data_output/vcfs/deepvariant_sample1_t.vcf /mnt/disks/data/data_output/vcfs/deepvariant_sample1_n.vcf


#running cnvkit
pbrun cnvkit --cnvkit-options="--count-reads --drop-low-coverage" --ref data/Ref/hg19_ref_genome.fasta --in-bam data/bam_output/XXXXXXX_180115_XXXXXXX_Homo-sapiens_normal_sample1.bam --out-file data/CNA/tumor1_cnvkit.vcf &>/data/logs/sample1_cnvkit_normal.log

pbrun cnvkit --cnvkit-options="--count-reads --drop-low-coverage" --ref data/Ref/hg19_ref_genome.fasta --in-bam data/bam_output/XXXXXXX_180115_XXXXXXX_Homo-sapiens_tumor_sample1.bam --out-file data/CNA/tumor1_cnvkit.vcf &>/data/logs/sample1_cnvkit_normal.log

#clearing disk cache after runs

sync
echo 1 | sudo tee /proc/sys/vm/drop_caches
echo 2 | sudo tee /proc/sys/vm/drop_caches
echo 3 | sudo tee /proc/sys/vm/drop_caches
