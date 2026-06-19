import os
import re
import csv
import logging
from pathlib import Path

# 设置日志格式
logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')

def get_target_atoms(gjf_path):
    """
    从 .gjf 文件中解析 C 和 B 的原子编号
    """
    c_idx = None
    b_idx = None
    atom_map = {}
    
    with open(gjf_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    in_molecule_spec = False
    atom_counter = 1
    
    for line in lines:
        line_s = line.strip()
        
        # 寻找分子说明部分
        if not in_molecule_spec and len(atom_map) == 0 and re.match(r'^-?\d+\s+\d+$', line_s):
            in_molecule_spec = True
            continue
            
        if in_molecule_spec:
            if line_s == '':
                in_molecule_spec = False 
                continue
            parts = line_s.split()
            if len(parts) >= 4 and parts[0].isalpha():
                atom_map[atom_counter] = parts[0].upper()
                atom_counter += 1
                
        # 寻找键的约束
        elif len(atom_map) > 0 and line_s != '':
            m = re.match(r'^(?:[A-Za-z]\s+)?(\d+)\s+(\d+)(?:\s+[A-Za-z]+)?$', line_s)
            if m:
                idx1 = int(m.group(1))
                idx2 = int(m.group(2))
                if idx1 in atom_map and idx2 in atom_map:
                    sym1 = atom_map[idx1]
                    sym2 = atom_map[idx2]
                    if (sym1 == 'C' and sym2 == 'B') or (sym1 == 'B' and sym2 == 'C'):
                        c_idx = idx1 if sym1 == 'C' else idx2
                        b_idx = idx1 if sym1 == 'B' else idx2
                        break

    return c_idx, b_idx

def process_nbo_log(log_path):
    """
    处理单个 NBO .log 文件，提取所有 C-B 键数据和原文
    返回: (all_found_bonds 列表, file_warnings 列表)
    """
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
        
    normal_term = False
    has_bond_orb = False
    has_perturbation = False
    
    for line in lines[-100:]:
        if "Normal termination" in line:
            normal_term = True
            break
            
    for line in lines:
        if "Bond orbital/ Coefficients/ Hybrids" in line or "NATURAL BOND ORBITAL ANALYSIS" in line:
            has_bond_orb = True
        if "Second Order Perturbation Theory Analysis of Fock Matrix" in line:
            has_perturbation = True
            
    file_warnings = []
    if not normal_term or not has_bond_orb or not has_perturbation:
        file_warnings.append("文件未正常完成或缺少NBO分析内容")
        
    all_found_bonds = []
    in_bond_section = False
    
    # 匹配所有的 BD 键
    bd_pattern = re.compile(r"BD\s*\(\s*\d+\)\s+([A-Z][a-z]?)\s+(\d+)\s+-\s+([A-Z][a-z]?)\s+(\d+)\b")
    comp_pattern = re.compile(r"\(\s*([\d\.]+)%\).*?(C|B)\s+(\d+)\s+s\(\s*([\d\.]+)%\)p.*?\(\s*([\d\.]+)%\)d.*?\(\s*([\d\.]+)%\)")
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        if "NATURAL BOND ORBITAL ANALYSIS:" in line or "Bond orbital/ Coefficients/ Hybrids" in line:
            in_bond_section = True
        elif "NBO Directionality and" in line or "Second Order Perturbation" in line:
            in_bond_section = False
            
        if in_bond_section:
            m_bd = bd_pattern.search(line)
            if m_bd:
                sym1, idx1, sym2, idx2 = m_bd.groups()
                idx1, idx2 = int(idx1), int(idx2)
                
                # 如果这是一个 C-B 键
                if (sym1 == 'C' and sym2 == 'B') or (sym1 == 'B' and sym2 == 'C'):
                    c_idx = idx1 if sym1 == 'C' else idx2
                    b_idx = idx1 if sym1 == 'B' else idx2
                    
                    bond_dict = {
                        'c_idx': c_idx,
                        'b_idx': b_idx,
                        'data': {},
                        'bond_raw': [line.rstrip('\n')],
                        'perturb_raw': []
                    }
                    
                    j = i + 1
                    while j < len(lines):
                        sub_line = lines[j]
                        # 遇到数字开头加上括号说明新的轨道开始了，比如 1. (
                        if re.match(r'^\s*\d+\.\s*\(', sub_line): 
                            break 
                        
                        bond_dict['bond_raw'].append(sub_line.rstrip('\n'))
                        
                        m_comp = comp_pattern.search(sub_line)
                        if m_comp:
                            pol = m_comp.group(1)
                            atom_sym = m_comp.group(2)
                            atom_idx = int(m_comp.group(3))
                            s_pct = m_comp.group(4)
                            p_pct = m_comp.group(5)
                            d_pct = m_comp.group(6)
                            
                            if atom_sym == 'C' and atom_idx == c_idx:
                                bond_dict['data']['C_pol'] = pol
                                bond_dict['data']['C_s'] = s_pct
                                bond_dict['data']['C_p'] = p_pct
                                bond_dict['data']['C_d'] = d_pct
                            elif atom_sym == 'B' and atom_idx == b_idx:
                                bond_dict['data']['B_pol'] = pol
                                bond_dict['data']['B_s'] = s_pct
                                bond_dict['data']['B_p'] = p_pct
                                bond_dict['data']['B_d'] = d_pct
                        j += 1
                    i = j - 1 
                    
                    if 'C_pol' in bond_dict['data'] and 'B_pol' in bond_dict['data']:
                        all_found_bonds.append(bond_dict)
        i += 1
        
    # 二阶微扰部分
    in_perturb_section = False
    perturb_header = []
    
    i = 0
    while i < len(lines):
        line = lines[i]
        if "Second Order Perturbation Theory Analysis of Fock Matrix in NBO Basis" in line:
            in_perturb_section = True
            perturb_header = [line.rstrip('\n')]
            for offset in range(1, 6):
                if i + offset < len(lines):
                    perturb_header.append(lines[i+offset].rstrip('\n'))
            i += 5
            
            # 为每个记录的键添加表头
            for b in all_found_bonds:
                b['perturb_raw'].extend(perturb_header)
            continue
            
        if in_perturb_section:
            if "NATURAL LOCALIZED MOLECULAR ORBITAL" in line or "NBO summary" in line or line.strip() == "1|1|UNPC" or "Job cpu time" in line:
                in_perturb_section = False
            else:
                for b in all_found_bonds:
                    b_idx = b['b_idx']
                    # 只抓取 Acceptor 含有对应硼原子的行
                    if re.search(rf"\/.*\bB\s+{b_idx}\b", line):
                        b['perturb_raw'].append(line.rstrip('\n'))
        i += 1
        
    return all_found_bonds, file_warnings

def main():
    base_dir = Path(".")
    atoms_dir = base_dir / "Atoms"
    
    if not atoms_dir.exists():
        logging.error(f"当前目录下未找到 Atoms 文件夹 ({atoms_dir.resolve()})")
        return

    csv_data = []
    all_log_output = []
    warning_data = []
    
    log_paths = [p for p in base_dir.rglob("*") if p.suffix.lower() == '.log' and "Atoms" not in p.parts and "1" not in p.parts]
    
    def natural_keys(text):
        return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', str(text))]
        
    log_paths.sort(key=lambda p: natural_keys(p.parent.name))
    
    for log_path in log_paths:
        structure_name = log_path.parent.name
        
        gjf_dir = atoms_dir / structure_name
        gjf_file = None
        if gjf_dir.exists():
            for f in gjf_dir.glob("*.gjf"):
                gjf_file = f
                break
                
        c_idx, b_idx = None, None
        if not gjf_file:
            msg = f"未找到对应的 .gjf 文件"
            warning_data.append({'Structure': structure_name, 'File': log_path.name, 'Warning': msg})
        else:
            c_idx, b_idx = get_target_atoms(gjf_file)
            if c_idx is None or b_idx is None:
                msg = f"无法从 {gjf_file.name} 中提取出 C 和 B 的原子约束编号"
                warning_data.append({'Structure': structure_name, 'File': log_path.name, 'Warning': msg})
                
        # 扫描整个日志，找到所有的 C-B 键
        all_found_bonds, file_warnings = process_nbo_log(log_path)
        
        if file_warnings:
            for w in file_warnings:
                warning_data.append({'Structure': structure_name, 'File': log_path.name, 'Warning': w})
                
        selected_bonds = []
        
        if c_idx is not None and b_idx is not None:
            # 尝试在找到的所有 C-B 键中精确匹配目标编号
            matched = [b for b in all_found_bonds if (b['c_idx'] == c_idx and b['b_idx'] == b_idx)]
            if matched:
                selected_bonds = matched
            else:
                msg = f"gjf中要求 C:{c_idx} B:{b_idx}，但在 log 中未找到对应的 C-B 键，自动回退到提取所有 C-B 键"
                warning_data.append({'Structure': structure_name, 'File': log_path.name, 'Warning': msg})
                selected_bonds = all_found_bonds
        else:
            # 如果一开始就没有 C B 编号，就提取所有的 C-B 键
            selected_bonds = all_found_bonds
            
        if not selected_bonds:
            msg = "在 log 文件中未能提取到任何 C-B 键数据"
            warning_data.append({'Structure': structure_name, 'File': log_path.name, 'Warning': msg})
            continue
            
        # 写入收集的数据
        for idx, b in enumerate(selected_bonds):
            suffix = f"_{idx+1}" if len(selected_bonds) > 1 else ""
            struct_display = f"{structure_name}{suffix}"
            
            csv_data.append({
                'Structure': struct_display,
                'C_Polarization': b['data']['C_pol'],
                'B_Polarization': b['data']['B_pol'],
                'C_s': b['data']['C_s'],
                'C_p': b['data']['C_p'],
                'C_d': b['data']['C_d'],
                'B_s': b['data']['B_s'],
                'B_p': b['data']['B_p'],
                'B_d': b['data']['B_d'],
            })
            
            all_log_output.append(f"====== Structure: {struct_display} | C:{b['c_idx']} B:{b['b_idx']} | File: {log_path.name} ======")
            all_log_output.append("--- Bond orbital/ Coefficients/ Hybrids (C-B Bond) ---")
            all_log_output.extend(b['bond_raw'])
            all_log_output.append("\n--- Second Order Perturbation Theory Analysis (Acceptor on B) ---")
            all_log_output.extend(b['perturb_raw'])
            all_log_output.append("\n\n")
            
    # 输出结果文件
    csv_file_path = base_dir / "extracted_CB_bonds.csv"
    with open(csv_file_path, "w", newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'Structure', 'C_Polarization', 'B_Polarization', 
            'C_s', 'C_p', 'C_d', 'B_s', 'B_p', 'B_d'
        ])
        writer.writeheader()
        writer.writerows(csv_data)
        
    log_file_path = base_dir / "extracted_raw_nbo.log"
    with open(log_file_path, "w", encoding='utf-8') as f:
        f.write("\n".join(all_log_output))
        
    warnings_csv_path = base_dir / "warnings.csv"
    if warning_data:
        # 使用 utf-8-sig 防止 Excel 打开乱码
        with open(warnings_csv_path, "w", newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=['Structure', 'File', 'Warning'])
            writer.writeheader()
            writer.writerows(warning_data)
            
    print(f"数据抓取完成！结果已保存至:")
    print(f"CSV文件: {csv_file_path.absolute()}")
    print(f"Log文件: {log_file_path.absolute()}")
    if warning_data:
        print(f"警告信息已汇总至: {warnings_csv_path.absolute()}")

if __name__ == "__main__":
    main()
