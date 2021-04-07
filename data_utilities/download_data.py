import requests 
from tqdm import tqdm
file_url = "https://d307kc0mrhucc3.cloudfront.net/train_images.tgz?Expires=1613610133&Signature=RXMiTYa4KDt4CA5lBnvzfGbzla9QZ-R2HWMOXXc7IZUJDRB1dM2GJ7nvH7VmrTSEpgLaCItP9rIujcmz2sVlmBIQLeSwNbMZElXBowY~LYFun2MXcI9DbLGNb9cCgoZakb4TV1nShNxk0a0VVIOGrGBSrC3HRxThwpFiVwvmtoizlKUtdR9WBAl7dMPrz4EzMFDoyEMsDPgX~7IkCO0Rq6b8ZqhVG1IpF7w0AK4zOSgAtdGQw6cnO9bWIMmFsTn5HHcmQHBZPVnxM~~Vd46rNFVO0vtJkAnI7axYuS0DG7h7CZgvNYs8gWg6K5-2QLc32BsrNq~zHn3CZ0U2cfI5QQ__&Key-Pair-Id=APKAIKGDJB5C3XUL2DXQ"
	
r = requests.get(file_url, stream = True) 

with open(work_dir+data_dir+'train_images.tgz', "wb") as file: 
	for block in tqdm(r.iter_content(chunk_size = 1024)): 
		if block: 
			file.write(block) 