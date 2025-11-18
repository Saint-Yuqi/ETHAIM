from sentence_transformers import SentenceTransformer

def main():
    # 1. 加载预训练模型
    # 第一次执行时，会从 Hugging Face 自动下载权重到本地缓存 (~/.cache/huggingface)
    print("Loading model intfloat/multilingual-e5-base ...")
    model = SentenceTransformer("intfloat/multilingual-e5-base")
    print("Model loaded.")

    # 2. 准备几句测试文本（注意加上 E5 推荐的前缀）
    sentences = [
        "query: airpods pro",
        "query: kinder spielzeug",
        "passage: Apple AirPods Pro (2nd generation) | Apple | In-Ear Headphones",
    ]

    # 3. 计算句向量
    embeddings = model.encode(sentences)

    print("Embeddings shape:", embeddings.shape)
    for i, sent in enumerate(sentences):
        print(f"{i}: {sent}")
        print(f"   embedding[:5] = {embeddings[i][:5]}")  # 只打印前 5 个维度

if __name__ == "__main__":
    main()
