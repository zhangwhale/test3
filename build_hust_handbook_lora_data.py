import re
import json
from pathlib import Path
from pypdf import PdfReader

pdf_path = Path("2025研究生手册.pdf")
out_path = Path("dataset/hust_graduate_handbook_lora.jsonl")

if not pdf_path.exists():
    raise FileNotFoundError(f"未找到 {pdf_path}，请先把《2025研究生手册.pdf》放到 minimind 根目录。")

reader = PdfReader(str(pdf_path))

pages = []
for i, page in enumerate(reader.pages):
    text = page.extract_text() or ""
    text = text.replace("\u3000", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    pages.append((i + 1, text.strip()))

full_text = "\n".join(text for _, text in pages)

# 按“第X条”切分条款
article_pattern = re.compile(r"(第[一二三四五六七八九十百零〇\d]+条[\s\S]*?)(?=\n第[一二三四五六七八九十百零〇\d]+条|\n第[一二三四五六七八九十]+章|$)")
articles = article_pattern.findall(full_text)

samples = []

system_prompt = (
    "你是华中科技大学研究生手册问答助手。"
    "回答问题时应依据《华中科技大学研究生手册》内容，表达准确、条理清晰。"
    "如果问题涉及具体流程、条件或时间要求，应分点说明。"
)

def clean_text(s: str) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    s = s.replace(" 。", "。").replace(" ，", "，")
    return s

# 1. 自动条款型问答
for art in articles:
    art = clean_text(art)
    if len(art) < 80:
        continue
    if len(art) > 1200:
        art = art[:1200] + "……"

    m = re.match(r"(第[一二三四五六七八九十百零〇\d]+条)", art)
    if not m:
        continue

    article_no = m.group(1)
    preview = art[:80]

    user_q = f"请根据《华中科技大学研究生手册》，说明{article_no}的主要内容。"
    assistant_a = art

    samples.append({
        "conversations": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_q},
            {"role": "assistant", "content": assistant_a}
        ]
    })

    user_q2 = f"《华中科技大学研究生手册》中，关于“{preview}”这部分内容是如何规定的？"
    samples.append({
        "conversations": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_q2},
            {"role": "assistant", "content": assistant_a}
        ]
    })

# 2. 人工增强的重点问答，覆盖常见提问
manual_qa = [
    (
        "硕士研究生的开题答辩一般什么时候完成？",
        "根据《华中科技大学硕士研究生培养工作规定》，硕士生应在导师指导下及时确定研究方向和课题，完成开题报告。开题答辩一般应在第二学期结束前进行，最迟不晚于第三学期结束前完成。开题答辩不通过的，三个月后方可再次申请开题答辩。"
    ),
    (
        "博士研究生的开题答辩有什么时间要求？",
        "根据《华中科技大学博士研究生培养工作规定》，博士生应在导师指导下完成文献综述、确定研究方向和课题，并进行开题答辩。直博生开题答辩一般于第五学期完成，最迟于第六学期完成；其他博士生开题答辩一般于第三学期完成，最迟于第四学期完成。开题答辩不通过的，三个月后方可再次申请。"
    ),
    (
        "硕士研究生中期检查主要检查什么内容？",
        "硕士研究生中期检查内容主要包括思想品德表现、课程学习完成情况、科研或实践中期进展情况三个方面。对于专业学位硕士生，还应按照专业学位研究生专业实践相关管理规定，对专业实践完成情况进行检查。"
    ),
    (
        "博士研究生中期考核不合格会怎样？",
        "博士研究生中期考核不合格的，三个月后方可再次参加中期考核。中期考核总次数不得超过三次。三次不合格者，不宜继续攻读博士学位，应终止博士阶段学业。直博生和硕博连读生可申请转为硕士生培养、申请结业或办理退学；其他博士生可申请结业或办理退学。"
    ),
    (
        "研究生个人培养计划应该什么时候完成？",
        "研究生应在导师指导下按照培养方案制定个人培养计划。个人培养计划一般应于入学后一个月内完成。如需修改，须经导师同意并报培养单位备案。"
    ),
    (
        "专业学位研究生专业实践的目的是什么？",
        "专业学位研究生专业实践的目的是培养专业学位研究生的实践研究和创新能力，使其熟悉行业工作流程和职业技术规范。专业实践是专业学位研究生培养的必修环节，研究生须按照培养计划开展专业实践，获得相应学分后，方能进入学位论文或实践成果答辩环节。"
    ),
    (
        "专业学位研究生专业实践一般需要多长时间？",
        "专业学位研究生专业实践的学分设置和时长应根据国家相应专业学位教育指导委员会指导性培养方案执行，由院系在培养方案中明确。一般应累计不少于半年，不具备工作经历的研究生原则上不少于一年。"
    ),
    (
        "研究生注册需要满足哪些基本条件？",
        "学生办理注册手续通常需要满足以下条件：入学或学习资格得到学校认可；处于学校规定的学习期限内；在学校规定的注册时间内；按学校规定缴纳学费及有关费用。符合条件并完成注册后，学生获得该学期学习资格。"
    ),
    (
        "研究生课程考核分为哪几种方式？",
        "研究生课程考核方式分为考试和考查两种。学位课程采用考试方式，其他课程可以采用考试方式或考查方式，研究环节采用考查方式。课程考核的具体形式、成绩构成及考核标准由院系结合实际情况制定，并写入课程教学大纲。"
    ),
    (
        "研究生提前毕业需要满足哪些基本条件？",
        "研究生申请提前毕业一般需要满足：在读期间遵纪守法、道德品质良好、恪守学术规范；已按培养方案修满应修学分并完成必修环节，学业成绩优良；研究成果突出，达到院系提前毕业要求；完成学位论文；博士生预答辩合格；已缴清在读期间学费。"
    )
]

for q, a in manual_qa:
    samples.append({
        "conversations": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": q},
            {"role": "assistant", "content": a}
        ]
    })

# 去重
seen = set()
dedup = []
for s in samples:
    key = s["conversations"][1]["content"]
    if key not in seen:
        seen.add(key)
        dedup.append(s)

out_path.parent.mkdir(parents=True, exist_ok=True)

with out_path.open("w", encoding="utf-8") as f:
    for item in dedup:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"已生成 LoRA 微调数据集：{out_path}")
print(f"样本数量：{len(dedup)}")
