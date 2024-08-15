## 数据处理
```bash
conda activate test1
# 复制原始对齐数据
cp /mnt/cfs/SPEECH/hupeng/align_dir/data_info/magic/train/data.list /home/liuhuang/workspace/tal_data_tools/data/filter_hupeng_align_magic_train_data.list

# 提取并过滤数据(小于8个字符的丢弃)
cd /home/liuhuang/workspace/tal_data_tools
# 注意修改一些参数，例如需要过滤的文件和保存路径，还有是否分段保存(默认分段10000条数据保存)
python filter_short_form_align_data.py
```