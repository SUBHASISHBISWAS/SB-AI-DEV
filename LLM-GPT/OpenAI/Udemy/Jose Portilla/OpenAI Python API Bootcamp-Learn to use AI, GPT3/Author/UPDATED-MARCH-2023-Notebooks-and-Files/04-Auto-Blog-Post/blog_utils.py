from pathlib import Path
import shutil
import os

from bs4 import BeautifulSoup as Soup
from git import Repo


def create_new_blog(path_to_content, title, content, cover_image=Path("../PT Centered Purple.png")):
    
    cover_image = Path(cover_image)
    
    files =len(list(path_to_content.glob("*.html")))
    new_title = f"{files+1}.html"
    path_to_new_content = path_to_content/new_title
    
    shutil.copy(cover_image, path_to_content)
    if not os.path.exists(path_to_new_content):
        with open(path_to_new_content, "w") as f:
            f.write("<!DOCTYPE html>\n")
            f.write("<html>\n")
            f.write("<head>\n")
            f.write(f"<title> {title} </title>\n")
            f.write("</head>\n")
            
            f.write("<body>\n")
            f.write(f"<img src='{cover_image.name}' alt='Cover Image'> <br />\n")
            f.write(f"<h1> {title} </h1>")
            f.write(content.replace("\n", "<br />\n"))
            f.write("</body>\n")
            f.write("</html>\n")
            print("Blog created")
            return path_to_new_content
    else:
        raise FileExistsError("File already exist! Abort")

def check_for_duplicate_links(path_to_new_content, links):
    urls = [str(link.get("href")) for link in links]
    content_path = str(Path(*path_to_new_content.parts[-2:]))
    return content_path in urls


def write_to_index(path_to_blog, path_to_new_content):
    with open(path_to_blog/"index.html") as index:
        soup = Soup(index.read())

    links = soup.find_all("a")
    last_link = links[-1]
    
    if check_for_duplicate_links(path_to_new_content, links):
        raise ValueError("Link does already exist!")
        
    link_to_new_blog = soup.new_tag("a", href=Path(*path_to_new_content.parts[-2:]))
    link_to_new_blog.string = path_to_new_content.name.split(".")[0]
    last_link.insert_after(link_to_new_blog)
    
    with open(path_to_blog/"index.html", "w") as f:
        f.write(str(soup.prettify(formatter='html')))


def update_blog(path_to_blog_repo, commit_message="Updated blog"):
    repo = Repo(path_to_blog_repo)
    repo.git.add(all=True)
    repo.index.commit(commit_message)
    origin = repo.remote(name='origin')
    origin.push()
