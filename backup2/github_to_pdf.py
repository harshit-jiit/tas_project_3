import os
import requests
import base64
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import HexColor
import argparse
from urllib.parse import urlparse
import mimetypes
from typing import List, Dict, Tuple

class GitHubToPDFConverter:
    def __init__(self, github_token: str = None):
        self.github_token = github_token
        self.session = requests.Session()
        if github_token:
            self.session.headers.update({'Authorization': f'token {github_token}'})
        
        # Supported file extensions
        self.supported_extensions = {
            '.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.hpp',
            '.css', '.html', '.xml', '.json', '.yaml', '.yml',
            '.md', '.txt', '.rst', '.sql', '.sh', '.bat',
            '.dockerfile', '.gitignore', '.env', '.ini', '.cfg', '.conf',
            '.go', '.rs', '.php', '.rb', '.swift', '.kt'
        }
        
        # Files to exclude (models, binaries, large files)
        self.exclude_patterns = {
            '.pkl', '.pickle', '.h5', '.hdf5', '.pt', '.pth', '.ckpt',
            '.model', '.bin', '.exe', '.dll', '.so', '.dylib',
            '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg', '.ico',
            '.mp4', '.avi', '.mov', '.wmv', '.flv', '.mp3', '.wav',
            '.zip', '.tar', '.gz', '.rar', '.7z', '.pdf', '.doc', '.docx'
        }
        
        # Maximum file size in bytes (1MB)
        self.max_file_size = 1024 * 1024
        
    def parse_github_url(self, url: str) -> Tuple[str, str]:
        """Parse GitHub URL to extract owner and repository name."""
        # Handle both HTTPS and SSH URLs
        if url.startswith('git@github.com:'):
            # SSH format: git@github.com:owner/repo.git
            path_part = url.replace('git@github.com:', '')
        else:
            # HTTPS format
            parsed = urlparse(url)
            if 'github.com' not in parsed.netloc:
                raise ValueError("URL must be a GitHub repository URL")
            path_part = parsed.path.strip('/')
        
        path_parts = path_part.split('/')
        if len(path_parts) < 2:
            raise ValueError("Invalid GitHub repository URL")
        
        owner = path_parts[0]
        repo = path_parts[1]
        
        # Remove .git extension if present
        if repo.endswith('.git'):
            repo = repo[:-4]
        
        return owner, repo
    
    def get_repository_contents(self, owner: str, repo: str, path: str = "") -> List[Dict]:
        """Recursively fetch all files from the repository."""
        url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
        
        try:
            response = self.session.get(url)
            response.raise_for_status()
            contents = response.json()
            
            files = []
            for item in contents:
                if item['type'] == 'file':
                    # Check file extension and size
                    file_ext = os.path.splitext(item['name'])[1].lower()
                    if (file_ext in self.supported_extensions and 
                        file_ext not in self.exclude_patterns and
                        item['size'] <= self.max_file_size):
                        files.append(item)
                elif item['type'] == 'dir' and not item['name'].startswith('.'):
                    # Recursively get contents of subdirectories
                    subdir_files = self.get_repository_contents(owner, repo, item['path'])
                    files.extend(subdir_files)
            
            return files
            
        except requests.RequestException as e:
            print(f"Error fetching repository contents: {e}")
            return []
    
    def get_file_content(self, file_info: Dict) -> str:
        """Download and decode file content."""
        try:
            response = self.session.get(file_info['download_url'])
            response.raise_for_status()
            
            # Try to decode as text
            try:
                return response.content.decode('utf-8')
            except UnicodeDecodeError:
                try:
                    return response.content.decode('latin-1')
                except UnicodeDecodeError:
                    return f"[Binary file or encoding error: {file_info['name']}]"
                    
        except requests.RequestException as e:
            return f"[Error downloading file: {e}]"
    
    def setup_pdf_styles(self):
        """Setup PDF styles for different content types."""
        styles = getSampleStyleSheet()
        
        # Custom styles
        code_style = ParagraphStyle(
            'CodeStyle',
            parent=styles['Code'],
            fontName='Courier',
            fontSize=8,
            leading=10,
            leftIndent=0.2*inch,
            rightIndent=0.2*inch,
            spaceAfter=6,
            borderColor=HexColor('#CCCCCC'),
            borderWidth=1,
            borderPadding=5,
            backColor=HexColor('#F8F8F8')
        )
        
        filename_style = ParagraphStyle(
            'FilenameStyle',
            parent=styles['Heading2'],
            fontSize=12,
            spaceAfter=6,
            spaceBefore=12,
            textColor=HexColor('#2E8B57'),
            borderColor=HexColor('#2E8B57'),
            borderWidth=1,
            borderPadding=3
        )
        
        path_style = ParagraphStyle(
            'PathStyle',
            parent=styles['Normal'],
            fontSize=9,
            textColor=HexColor('#666666'),
            fontName='Courier',
            spaceAfter=6
        )
        
        return {
            'code': code_style,
            'filename': filename_style,
            'path': path_style,
            'normal': styles['Normal'],
            'title': styles['Title']
        }
    
    def escape_html(self, text: str) -> str:
        """Escape HTML characters for ReportLab."""
        return (text.replace('&', '&amp;')
                   .replace('<', '&lt;')
                   .replace('>', '&gt;')
                   .replace('\n', '<br/>'))
    
    def create_pdf(self, files_data: List[Tuple[Dict, str]], output_path: str, repo_info: str):
        """Create PDF document with all the code files."""
        doc = SimpleDocTemplate(output_path, pagesize=A4)
        story = []
        styles = self.setup_pdf_styles()
        
        # Title page
        title = Paragraph(f"Source Code Repository: {repo_info}", styles['title'])
        story.append(title)
        story.append(Spacer(1, 0.5*inch))
        
        # Summary
        summary_text = f"Generated PDF containing {len(files_data)} source files from GitHub repository."
        summary = Paragraph(summary_text, styles['normal'])
        story.append(summary)
        story.append(PageBreak())
        
        # Add each file
        for i, (file_info, content) in enumerate(files_data):
            print(f"Processing file {i+1}/{len(files_data)}: {file_info['path']}")
            
            # File header
            filename = Paragraph(f"File: {file_info['name']}", styles['filename'])
            story.append(filename)
            
            filepath = Paragraph(f"Path: {file_info['path']}", styles['path'])
            story.append(filepath)
            
            story.append(Spacer(1, 0.1*inch))
            
            # File content
            if content.strip():
                # Split content into manageable chunks to avoid memory issues
                lines = content.split('\n')
                chunk_size = 100  # Process 100 lines at a time
                
                for chunk_start in range(0, len(lines), chunk_size):
                    chunk_end = min(chunk_start + chunk_size, len(lines))
                    chunk_lines = lines[chunk_start:chunk_end]
                    chunk_content = '\n'.join(chunk_lines)
                    
                    escaped_content = self.escape_html(chunk_content)
                    code_para = Paragraph(escaped_content, styles['code'])
                    story.append(code_para)
            else:
                empty_file = Paragraph("[Empty file]", styles['code'])
                story.append(empty_file)
            
            # Add page break between files (except for the last file)
            if i < len(files_data) - 1:
                story.append(PageBreak())
        
        # Build PDF
        print("Building PDF...")
        doc.build(story)
        print(f"PDF created successfully: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Convert GitHub repository to PDF')
    parser.add_argument('repo_url', help='GitHub repository URL')
    parser.add_argument('-o', '--output', default='repository_code.pdf', help='Output PDF filename')
    parser.add_argument('-t', '--token', help='GitHub API token (optional, for private repos or higher rate limits)')
    
    args = parser.parse_args()
    
    # Create converter
    converter = GitHubToPDFConverter(args.token)
    
    try:
        # Parse GitHub URL
        owner, repo = converter.parse_github_url(args.repo_url)
        repo_info = f"{owner}/{repo}"
        print(f"Processing repository: {repo_info}")
        
        # Get repository contents
        print("Fetching repository contents...")
        files = converter.get_repository_contents(owner, repo)
        
        if not files:
            print("No supported files found in the repository.")
            return
        
        print(f"Found {len(files)} supported files")
        
        # Download file contents
        files_data = []
        for i, file_info in enumerate(files):
            print(f"Downloading {i+1}/{len(files)}: {file_info['path']}")
            content = converter.get_file_content(file_info)
            files_data.append((file_info, content))
        
        # Create PDF
        converter.create_pdf(files_data, args.output, repo_info)
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Example usage if run directly
    if len(os.sys.argv) == 1:
        print("Example usage:")
        print("python github_to_pdf.py https://github.com/username/repository")
        print("python github_to_pdf.py https://github.com/username/repository -o my_repo.pdf")
        print("python github_to_pdf.py https://github.com/username/repository -t your_github_token")
    else:
        main()