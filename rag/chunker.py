from pathlib import Path


def load_document(file_path):
    path = Path(file_path)

    return path.read_text(encoding="utf-8")


def chunk_markdown(text):
    sections = text.split("## ")

    chunks = []

    for index, section in enumerate(sections):
        section = section.strip()

        if not section:
            continue

        lines = section.split("\n")

        title = lines[0].strip()
        content = "\n".join(lines[1:]).strip()

        if content:
            chunks.append({
                "chunk_id": f"REF_{index:03d}",
                "concept": "references",
                "title": title,
                "content_type": get_content_type(title),
                "content": content
            })

    return chunks


def get_content_type(title):
    title = title.lower()

    if "misconception" in title:
        return "misconception"

    if "worked example" in title:
        return "worked_example"

    if "assignment" in title:
        return "concept"

    if "mutable" in title:
        return "concept"

    return "concept"


if __name__ == "__main__":
    document = load_document(
        "knowledge/references.md"
    )

    chunks = chunk_markdown(document)

    for i, chunk in enumerate(chunks):
        print(f"\n--- Chunk {i + 1} ---")
        print("Title:", chunk["title"])
        print("Content:", chunk["content"])
