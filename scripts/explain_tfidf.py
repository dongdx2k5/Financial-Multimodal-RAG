"""Bài thực hành TF-IDF chỉ đọc: không gọi Ollama, không sửa dữ liệu dự án.

Chạy từ thư mục dự án:
    .venv\\Scripts\\python.exe scripts\\explain_tfidf.py --mode simplified --verify
    .venv\\Scripts\\python.exe scripts\\explain_tfidf.py --mode project --verify

Đây là dữ liệu GIẢ LẬP để học phép tính, không phải số liệu báo cáo tài chính.
Demo chỉ tách riêng điểm từ vựng TF-IDF. Nó không chạy expand_query, cộng điểm
graph, lọc document_id hay các quy tắc xếp hạng khác của HybridRetriever.search.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from collections.abc import Sequence

import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

TOY_CHUNKS = (
    "doanh thu doanh thu tang",
    "loi nhuan tang",
    "chi phi tang",
)
DEFAULT_QUERY = "doanh thu"


def build_vectorizer(mode: str = "simplified") -> TfidfVectorizer:
    """Tạo cấu hình đơn giản hoặc đúng cấu hình TF-IDF của dự án chính."""
    if mode not in {"simplified", "project"}:
        raise ValueError("mode phải là 'simplified' hoặc 'project'")
    return TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 1) if mode == "simplified" else (1, 2),
        max_features=50_000,
        token_pattern=r"(?u)\b\w+\b",
    )


def normalize_l2(rows: np.ndarray) -> np.ndarray:
    """Chia từng hàng cho độ dài L2; vector toàn 0 vẫn giữ nguyên toàn 0."""
    lengths = np.linalg.norm(rows, axis=1, keepdims=True)
    return np.divide(rows, lengths, out=np.zeros_like(rows), where=lengths != 0)


def explain_tfidf(
    chunks: Sequence[str] = TOY_CHUNKS,
    query: str = DEFAULT_QUERY,
    mode: str = "simplified",
) -> dict:
    """Tính từng bước bằng NumPy, đồng thời giữ kết quả thật từ scikit-learn.

    Ma trận được đổi sang dense chỉ vì ví dụ có 3 đoạn rất nhỏ. Dự án thật
    giữ ma trận sparse để tiết kiệm bộ nhớ.
    """
    vectorizer = build_vectorizer(mode)
    sparse_matrix = vectorizer.fit_transform(chunks)
    # Không fit lại câu hỏi: dùng từ điển và IDF đã học từ các đoạn tài liệu.
    sparse_query = vectorizer.transform([query])
    features = vectorizer.get_feature_names_out()
    analyzer = vectorizer.build_analyzer()
    token_lists = [analyzer(chunk) for chunk in chunks]
    counters = [Counter(tokens) for tokens in token_lists]
    query_tokens = analyzer(query)
    query_counter = Counter(query_tokens)

    # TF đang dùng là số lần xuất hiện thô, không chia cho độ dài đoạn.
    counts = np.array([[counter[word] for word in features] for counter in counters])
    query_counts = np.array([[query_counter[word] for word in features]])
    document_frequency = (counts > 0).sum(axis=0)
    idf = np.log((1 + len(chunks)) / (1 + document_frequency)) + 1
    raw_weights = counts * idf
    normalized = normalize_l2(raw_weights)
    query_raw_weights = query_counts * idf
    normalized_query = normalize_l2(query_raw_weights)
    # Sau chuẩn hóa L2, tích vô hướng là cosine similarity.
    scores = (sparse_matrix @ sparse_query.T).toarray().ravel()
    contributions = normalized * normalized_query

    return {
        "mode": mode,
        "chunks": tuple(chunks),
        "query": query,
        "vectorizer": vectorizer,
        "sparse_matrix": sparse_matrix,
        "sparse_query": sparse_query,
        "features": features,
        "tokens": token_lists,
        "query_tokens": query_tokens,
        "counts": counts,
        "query_counts": query_counts,
        "df": document_frequency,
        "idf": idf,
        "raw_weights": raw_weights,
        "normalized": normalized,
        "query_raw_weights": query_raw_weights,
        "normalized_query": normalized_query,
        "scores": scores,
        "contributions": contributions,
        "oov_tokens": sorted(set(query_tokens) - set(features)),
    }


def verify_explanation(result: dict) -> None:
    """Báo lỗi nếu cách tính tay không khớp thư viện; tuyệt đối không ghi file."""
    np.testing.assert_allclose(result["idf"], result["vectorizer"].idf_, atol=1e-12)
    np.testing.assert_allclose(
        result["normalized"], result["sparse_matrix"].toarray(), atol=1e-12
    )
    np.testing.assert_allclose(
        result["normalized_query"], result["sparse_query"].toarray(), atol=1e-12
    )
    cosine = cosine_similarity(result["sparse_matrix"], result["sparse_query"]).ravel()
    np.testing.assert_allclose(result["scores"], cosine, atol=1e-12)
    np.testing.assert_allclose(
        result["contributions"].sum(axis=1), result["scores"], atol=1e-12
    )


def print_explanation(result: dict) -> None:
    """In thứ tự bước tính để người mới có thể dò lại từng con số."""
    print("TF-IDF: DỮ LIỆU GIẢ LẬP, KHÔNG PHẢI SỐ LIỆU TÀI CHÍNH THẬT")
    print(f"scikit-learn: {sklearn.__version__}; chế độ: {result['mode']}")
    print("Chỉ minh họa điểm TF-IDF, chưa cộng graph/mở rộng câu hỏi/lọc tài liệu.")
    print("TF = số lần xuất hiện; IDF = ln((N + 1) / (df + 1)) + 1; norm = L2.")
    print(f"\n1. N = {len(result['chunks'])} đoạn tài liệu trong toàn bộ corpus ví dụ.")
    for i, (chunk, tokens) in enumerate(zip(result["chunks"], result["tokens"]), start=1):
        print(f"C{i}: {chunk}")
        print(f"    Các đặc trưng trước khi đếm: {tokens}")
    print(f"Câu hỏi: {result['query']}")
    print(f"Đặc trưng của câu hỏi: {result['query_tokens']}")

    print("\n2. Từ điển, số lần xuất hiện TF, số đoạn chứa từ df và IDF:")
    print("đặc trưng             TF(C1,C2,C3)       df        IDF")
    for j, feature in enumerate(result["features"]):
        tf = ",".join(str(int(value)) for value in result["counts"][:, j])
        print(f"{feature:<22}{tf:<18}{result['df'][j]:>4}  {result['idf'][j]:>10.6f}")
    print(f"Thứ tự cột vector: {list(result['features'])}")
    print(f"Kích thước ma trận: {result['sparse_matrix'].shape} = (số đoạn, số đặc trưng)")
    print(f"Kích thước vector câu hỏi: {result['sparse_query'].shape}")

    print("\n3. Vector của từng đoạn, trước và sau chuẩn hóa L2:")
    for i in range(len(result["chunks"])):
        print(f"C{i + 1}: TF * IDF = {np.round(result['raw_weights'][i], 6)}")
        print(f"    độ dài L2 = {np.linalg.norm(result['raw_weights'][i]):.6f}")
        print(f"    đã chuẩn hóa = {np.round(result['normalized'][i], 6)}")
    print("\n4. Câu hỏi dùng transform, giữ nguyên từ điển và IDF:")
    print(f"TF câu hỏi: {result['query_counts'][0]}")
    print(f"TF * IDF câu hỏi: {np.round(result['query_raw_weights'][0], 6)}")
    print(f"Vector câu hỏi chuẩn hóa: {np.round(result['normalized_query'][0], 6)}")
    print(f"Đặc trưng ngoài từ điển bị bỏ qua: {result['oov_tokens']}")

    print("\n5. Điểm cosine = tổng đóng góp của từng đặc trưng:")
    ranking = sorted(range(len(result["chunks"])), key=lambda i: (-result["scores"][i], i))
    for i in ranking:
        print(f"C{i + 1}: {result['scores'][i]:.6f}")
        nonzero = [
            (feature, result["contributions"][i, j])
            for j, feature in enumerate(result["features"])
            if result["contributions"][i, j] > 0
        ]
        for feature, contribution in sorted(nonzero, key=lambda item: (-item[1], item[0])):
            print(f"    {feature!r}: {contribution:.6f}")
        if not nonzero:
            print("    Không có đặc trưng chung nên điểm TF-IDF bằng 0.")
    print("Điểm này đo mức trùng đặc trưng có trọng số, không phải xác suất câu trả lời đúng.")

    oov_query = "zxqv"
    if oov_query not in result["vectorizer"].vocabulary_:
        oov_vector = result["vectorizer"].transform([oov_query])
        oov_scores = (result["sparse_matrix"] @ oov_vector.T).toarray().ravel()
        print(f"\n6. OOV: câu hỏi {oov_query!r} không có trong từ điển.")
        print(f"Vector có {oov_vector.nnz} phần tử khác 0; điểm các đoạn = {oov_scores}.")


def main() -> None:
    # Windows có thể dùng cp1258 cho pipe, không biểu diễn đủ tiếng Việt dạng NFC.
    # Chỉ đổi encoding stdout của CLI này; không đụng cấu hình hệ thống.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("simplified", "project"), default="simplified")
    parser.add_argument("--query", default=DEFAULT_QUERY, help="Câu hỏi thử nghiệm.")
    parser.add_argument("--verify", action="store_true", help="Kiểm tra phép tính tay với sklearn.")
    args = parser.parse_args()
    result = explain_tfidf(query=args.query, mode=args.mode)
    print_explanation(result)
    if args.verify:
        verify_explanation(result)
        print("\nKIỂM TRA THÀNH CÔNG: IDF, vector, cosine và tổng đóng góp đều khớp.")


if __name__ == "__main__":
    main()
