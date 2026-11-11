# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# Use the kagglehub client library to attach Kaggle resources like competitions, datasets, and models to your session
# Learn more about kagglehub: https://github.com/Kaggle/kagglehub/blob/main/README.md

import kagglehub
# kagglehub.dataset_download('<owner>/<dataset-slug>')

import pandas as pd, numpy as np, json, re, email
from datasets import load_dataset
from huggingface_hub import login, hf_hub_download, list_repo_files
from kaggle_secrets import UserSecretsClient

HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
login(token=HF_TOKEN)

SEED = 42
frames = []

def add(df, source, label, n):
    """Clean, sample n rows, and append to the global list."""
    df = df.dropna(subset=["text"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() >= 10]
    df["text"] = df["text"].str.slice(0, 20000)   # cap extreme lengths
    df = df.drop_duplicates(subset="text")        # exact duplicates only (MinHash comes later)
    df = df.sample(n=min(n, len(df)), random_state=SEED)
    out = pd.DataFrame({"text": df["text"], "label": label, "source": source})
    print(f"{source:22s} kept {len(out)} (label={label})")
    frames.append(out)

import os
for root, dirs, files in os.walk("/kaggle/input"):
    print(root, files[:5])

#Step 2: Positive sources

#Gandalf and Mosscap (small, start here)

g1 = load_dataset("Lakera/gandalf_ignore_instructions", split="train").to_pandas()
g2 = load_dataset("Lakera/gandalf_summarization", split="train").to_pandas()
print(g1.columns.tolist(), g2.columns.tolist())
add(pd.concat([g1[["text"]], g2[["text"]]]), "gandalf", 1, 2500)

m = load_dataset("Lakera/mosscap_prompt_injection", split="train").to_pandas()
print(m.columns.tolist())
add(m.rename(columns={"prompt": "text"}), "mosscap", 1, 1500)

p = hf_hub_download("qxcv/tensor-trust", "raw-data/v1/raw_dump_attacks.jsonl.bz2",
                    repo_type="dataset")
tt = pd.read_json(p, lines=True, compression="bz2")
print(tt.columns.tolist())          # look for the success column

p = hf_hub_download("qxcv/tensor-trust", "raw-data/v1/raw_dump_attacks.jsonl.bz2",
                    repo_type="dataset")
tt = pd.read_json(p, lines=True, compression="bz2")
print(tt.columns.tolist())          # look for the success column

tt = tt[(tt["output_is_access_granted"] == True) & (tt["is_self_attack"] == False)]
tt = tt[tt["attacker_input"].str.strip() != tt["access_code"].str.strip()]
add(tt.rename(columns={"attacker_input": "text"}), "tensor_trust", 1, 4000)

hp = load_dataset("hackaprompt/hackaprompt-dataset", split="train", token=HF_TOKEN).to_pandas()
print(hp.columns.tolist(), len(hp))
hp = hp[hp["correct"] == True]             # successful attempts only
add(hp.rename(columns={"user_input": "text"}), "hackaprompt", 1, 3500)

# Step 3: Negative sources

# Dolly and OASST1 (hard negatives with imperatives)
d = load_dataset("databricks/databricks-dolly-15k", split="train").to_pandas()
add(d.rename(columns={"instruction": "text"}), "dolly", 0, 2500)

o = load_dataset("OpenAssistant/oasst1", split="train").to_pandas()
o = o[(o["role"] == "prompter") & (o["lang"] == "en")]
add(o, "oasst1", 0, 2000)

import os
print(os.listdir("/kaggle/input/"))        # confirm the folder name
path = "/kaggle/input/datasets/wcukierski/enron-email-dataset/emails.csv"
en = pd.read_csv(path, nrows=60000)        # file is large; read a slice
def body(raw):
    try:
        msg = email.message_from_string(raw)
        return msg.get_payload() if not msg.is_multipart() else ""
    except Exception:
        return None
en["text"] = en["message"].map(body)
add(en, "enron", 0, 4000)

files = list_repo_files("McAuley-Lab/Amazon-Reviews-2023", repo_type="dataset")
print([f for f in files if "All_Beauty" in f or "Appliances" in f])   # check exact names

def read_head(path, n):
    rows = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= n: break
            rows.append(json.loads(line))
    return pd.DataFrame(rows)

parts = []
for cat in ["All_Beauty", "Appliances"]:
    rp = hf_hub_download("McAuley-Lab/Amazon-Reviews-2023",
                         f"raw/review_categories/{cat}.jsonl", repo_type="dataset")
    parts.append(read_head(rp, 15000)[["text"]])
add(pd.concat(parts), "amazon_reviews", 0, 4000)

mp = hf_hub_download("McAuley-Lab/Amazon-Reviews-2023",
                     "raw/meta_categories/meta_All_Beauty.jsonl", repo_type="dataset")
meta = read_head(mp, 15000)
meta["text"] = meta["description"].map(lambda x: " ".join(x) if isinstance(x, list) else None)
add(meta, "amazon_descriptions", 0, 1500)

#Step 4: Combine, check and save
df = pd.concat(frames, ignore_index=True)
print(df.groupby(["label", "source"]).size())
print(df["label"].value_counts())

df.to_parquet("/kaggle/working/pilot_raw.parquet", index=False)

manifest = {"seed": SEED, "date": str(pd.Timestamp.today().date()),
            "counts": df.groupby("source").size().to_dict()}
json.dump(manifest, open("/kaggle/working/data_manifest.json", "w"), indent=2)

