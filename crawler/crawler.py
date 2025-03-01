import requests
from bs4 import BeautifulSoup
import os
import time
from datetime import datetime, timedelta

# Đảm bảo thư mục data tồn tại
if not os.path.exists("data"):
    os.makedirs("data")

# Đọc link từ file unused.txt
def read_unused_links():
    if not os.path.exists("unused.txt"):
        return []
    with open("unused.txt", "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines() if line.strip() and "uet" in line.lower()]

# Đọc link đã dùng từ file used.txt
def read_used_links():
    if not os.path.exists("used.txt"):
        return set()
    with open("used.txt", "r", encoding="utf-8") as f:
        return set(line.strip() for line in f.readlines() if line.strip())

# Ghi danh sách link đã dùng vào used.txt
def write_used_links(used_links):
    with open("used.txt", "a", encoding="utf-8") as f:
        for link in used_links:
            f.write(link + "\n")

# Cập nhật file unused.txt
def update_unused_links(unused_links):
    with open("unused.txt", "w", encoding="utf-8") as f:
        for link in unused_links:
            f.write(link + "\n")

# Đọc ID cuối cùng từ file last_id.txt
def get_last_id():
    if not os.path.exists("last_id.txt"):
        return 0
    with open("last_id.txt", "r", encoding="utf-8") as f:
        return int(f.read().strip())

# Lưu ID cuối cùng vào file last_id.txt
def save_last_id(last_id):
    with open("last_id.txt", "w", encoding="utf-8") as f:
        f.write(str(last_id))

# Chuyển bảng thành văn bản
def table_to_text(table, table_id):
    lines = [f"Table {table_id}:"]
    for i, tr in enumerate(table.find_all("tr")):
        cells = [td.get_text(strip=True) for td in tr.find_all(["th", "td"])]
        if cells:
            lines.append(f"Row {i+1}: {' | '.join(cells)}")
    return "\n".join(lines)

# Crawl trang và giữ thứ tự nội dung
def crawl_page(url, doc_id, data_dir):
    try:
        response = requests.get(url, timeout=10, verify=False)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        # Loại bỏ thẻ <script>
        for script in soup(["script"]):
            script.decompose()
        # Loại bỏ thẻ có id="main-nav"
        main_nav = soup.find(id="main-nav")
        if main_nav:
            main_nav.decompose()  # Xóa thẻ và nội dung bên trong nó
        top_nav = soup.find(id="top-nav")
        if top_nav:
            top_nav.decompose()  # Xóa thẻ và nội dung bên trong nó
        bottom = soup.find(id="bottom")
        if bottom:
            bottom.decompose()  # Xóa thẻ và nội dung bên trong nó
        bottom_nav = soup.find(id="bottom-nav")
        if bottom_nav:
            bottom_nav.decompose()  # Xóa thẻ và nội dung bên trong nó
        page_heading = soup.find_all("div", {"class": "page-heading"})
        if page_heading:
            for ph in page_heading:
                ph.decompose()

        # Lấy tiêu đề trang
        title = soup.title.string.strip() if soup.title else "No Title"

        # Xử lý nội dung tuần tự
        # content_lines = []
        # table_count = 0
        # seen_texts = set()
        # for element in soup.body.find_all(recursive=False):
        #     if element.name == "table":
        #         table_count += 1
        #         table_text = table_to_text(element, table_count)
        #         if table_text not in seen_texts:
        #             content_lines.append(table_text)
        #             seen_texts.add(table_text)
        #     else:
        #         text = element.get_text(separator="\n", strip=True)
        #         if text and text not in seen_texts:
        #             content_lines.append(text)
        #             seen_texts.add(text)
        
        # Tìm phần tử có id="content"
        content_section = soup.find(id="content")
        if not content_section:
            print(f"Không tìm thấy phần tử id='content' trong {url}")
            content_lines = ["Không có nội dung trong id='content'"]
            links = []
        else:
            # Xử lý nội dung trong id="content" tuần tự
            content_lines = []
            table_count = 0
            seen_texts = set()
            for element in content_section.find_all(recursive=False):
                if element.name == "table":
                    table_count += 1
                    table_text = table_to_text(element, table_count)
                    if table_text not in seen_texts:
                        content_lines.append(table_text)
                        seen_texts.add(table_text)
                else:
                    text = element.get_text(separator="\n", strip=True)
                    if text and text not in seen_texts:
                        content_lines.append(text)
                        seen_texts.add(text)

        # Kết hợp nội dung
        full_content = f"URL: {url}\nTitle: {title}\n\n" + "\n\n".join(content_lines)

        # Lưu vào file
        file_path = f"{data_dir}/doc{doc_id}.txt"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(full_content)

        # Tìm link
        links = []
        for a_tag in content_section.find_all("a", href=True):
            href = a_tag["href"]
            if "courses.uet" in href.lower():
                continue
            if "facebook" in href.lower():
                continue
            if "youtube" in href.lower():
                continue
            if "tiktok" in href.lower():
                continue
            if "uet." in href.lower():
                if href.startswith("/"):
                    href = url.rstrip("/") + href
                elif not href.startswith("http"):
                    continue
                links.append(href)

        return list(set(links))

    except Exception as e:
        print(f"Lỗi khi crawl {url}: {e}")
        return []

# Hàm chính
def main():
    data_dir = '../data'
    unused_links = read_unused_links()
    if not unused_links:
        print("Không có link nào trong unused.txt chứa 'uet' để crawl.")
        return

    # Lấy ID cuối cùng
    current_id = get_last_id()
    used_links_set = read_used_links()
    new_links = []

    # Thiết lập thời gian giới hạn
    time_limit_minutes = 30  # Đổi lại thành 10 phút
    start_time = datetime.now()
    end_time = start_time + timedelta(minutes=time_limit_minutes)

    print(f"Bắt đầu crawl lúc {start_time}. Giới hạn thời gian: {time_limit_minutes} phút.")

    # Crawl hết tất cả link trong unused.txt
    while unused_links:
        if datetime.now() > end_time:
            print(f"Đã vượt quá {time_limit_minutes} phút. Dừng chương trình.")
            break

        current_id += 1
        current_url = unused_links.pop(0)
        if current_url in used_links_set:
            continue
        print(f"Đang crawl: {current_url} (ID: {current_id})")

        # Crawl dữ liệu
        found_links = crawl_page(current_url, current_id, data_dir)
        used_links_set.add(current_url)

        # Kiểm tra link mới với cả used.txt và unused.txt
        current_unused_links = set(read_unused_links())  # Đọc lại từ file unused.txt
        for link in found_links:
            if link not in used_links_set and link not in current_unused_links:
                new_links.append(link)
                unused_links.append(link)

        # Ghi link đã dùng
        write_used_links([current_url])
        # Cập nhật unused.txt
        update_unused_links(unused_links)
        # Lưu ID mới nhất
        save_last_id(current_id)

        file_path = f"data/doc{current_id}.txt"
        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            print(f"Đã crawl xong {current_url}. Dữ liệu lưu tại {file_path}")
        else:
            print(f"Không có dữ liệu crawl được từ {current_url}")
        print(f"Thời gian còn lại: {int((end_time - datetime.now()).total_seconds() / 60)} phút")

    print(f"Đã crawl xong. Tổng cộng tìm thấy {len(new_links)} link mới liên quan chứa 'uet'.")
    print(f"Kết thúc lúc {datetime.now()}")

if __name__ == "__main__":
    main()