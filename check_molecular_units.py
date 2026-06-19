import os
import re
import csv
from pathlib import Path

def process_log_for_units(log_path):
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
        
    units = []
    
    # 正则匹配 " Molecular unit  1  (C6H12O2)"
    unit_pattern = re.compile(r"Molecular\s+unit\s+(\d+)\s+\((.*?)\)", re.IGNORECASE)
    
    for line in lines:
        m = unit_pattern.search(line)
        if m:
            unit_num = int(m.group(1))
            formula = m.group(2).strip()
            units.append((unit_num, formula))
            
    # 因为在 NBO 输出中，Molecular unit 列表可能会在多个计算阶段重复打印
    # 我们用一个字典去重，保留最新/所有对应的编号
    unique_units = {}
    for num, form in units:
        unique_units[num] = form
        
    return unique_units

def main():
    base_dir = Path(".")
    csv_data = []
    
    # 按照上一个脚本同样的逻辑获取所有有效 log 文件并自然排序
    log_paths = [p for p in base_dir.rglob("*") if p.suffix.lower() == '.log' and "Atoms" not in p.parts and "1" not in p.parts]
    
    def natural_keys(text):
        return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', str(text))]
        
    log_paths.sort(key=lambda p: natural_keys(p.parent.name))
    
    for log_path in log_paths:
        structure_name = log_path.parent.name
        
        unique_units = process_log_for_units(log_path)
        
        fragment_count = len(unique_units)
        # 获取按照编号排序的所有化学式
        formulas = [unique_units[k] for k in sorted(unique_units.keys())]
        
        # 判断是否含有 C6H12O2 片段
        has_c6h12o2 = "C6H12O2" in formulas
        
        # 问题判定逻辑
        # ① 没有片段的化学式为 C6H12O2 的分子
        # ② 分子不是被分为两个片段的情况
        is_missing_c6h12o2 = not has_c6h12o2
        is_not_two_fragments = (fragment_count != 2)
        
        # 如果命中了上述任何一个条件，则记录为异常结构
        if is_missing_c6h12o2 or is_not_two_fragments:
            reasons = []
            if is_missing_c6h12o2:
                reasons.append("未包含 C6H12O2 片段")
            if is_not_two_fragments:
                reasons.append(f"片段数量为 {fragment_count} (不是 2 个)")
                
            csv_data.append({
                'Structure': structure_name,
                'File': log_path.name,
                'Fragment_Count': fragment_count,
                'Formulas': " | ".join(formulas),
                'Reason': " + ".join(reasons)
            })
            
    out_csv = base_dir / "abnormal_molecular_units.csv"
    # 使用 utf-8-sig 编码保证在 Windows Excel 中打开不会乱码
    with open(out_csv, "w", newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=['Structure', 'File', 'Fragment_Count', 'Formulas', 'Reason'])
        writer.writeheader()
        writer.writerows(csv_data)
        
    print(f"片段检查分析完成！")
    print(f"共发现 {len(csv_data)} 个不符合预期的结构（可能没有 C6H12O2 或片段数不为 2）。")
    print(f"详细情况已保存至: {out_csv.absolute()}")

if __name__ == "__main__":
    main()
