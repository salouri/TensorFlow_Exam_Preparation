# def download_file(filePath, url):
#     print('downloading file ' + filePath.split('/')[-1] + ' started...')
#     req = requests.get(url)
#     with open(filePath, 'wb') as f:
#         f.write(req.content)
#     print('download completed!')
import os


def get_size(start_path='.'):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)

    return total_size


def downloadFile(url, file_path, force_download=False):
    import os
    import requests, sys

    filename = file_path.split(os.sep)[-1]
    fileDir = file_path.rstrip(filename)
    if os.path.isdir(fileDir) and not os.path.exists(file_path) or force_download :
        print('Downloading file:"' + filename, '"...')
        with open(file_path, 'wb') as f:
            response = requests.get(url, stream=True)
            total = response.headers.get('content-length')
            if total is None:
                f.write(response.content)
            else:
                downloaded = 0
                total = int(total)
                for data in response.iter_content(chunk_size=max(int(total / 1000), 1024 * 1024)):
                    downloaded += len(data)
                    f.write(data)
                    done = int(50 * (downloaded / total))
                    sys.stdout.write('\r[{}{}]'.format('█' * done, '.' * (50 - done)))
                    sys.stdout.flush()
        sys.stdout.write('\n')
        print('\t---> File Download completed.')
    else:
        print('No download is needed!\n'
              'Check if file: "' + filename + '" already exists!')


def downloadExtract(url, zipFilePath: str):
    import os
    import requests, sys

    filename = zipFilePath.split(os.sep)[-1]
    fileDir = zipFilePath.rstrip('.zip')
    if not os.path.exists(zipFilePath) or os.path.isdir(fileDir):
        print('Downloading file:"' + filename, '"...')
        with open(zipFilePath, 'wb') as f:
            response = requests.get(url, stream=True)
            total = response.headers.get('content-length')
            if total is None:
                f.write(response.content)
            else:
                downloaded = 0
                total = int(total)
                for data in response.iter_content(chunk_size=max(int(total / 1000), 1024 * 1024)):
                    downloaded += len(data)
                    f.write(data)
                    done = int(50 * (downloaded / total))
                    sys.stdout.write('\r[{}{}]'.format('█' * done, '.' * (50 - done)))
                    sys.stdout.flush()
        sys.stdout.write('\n')
        print('\t---> File Download completed.')
    else:
        print('No download is needed!\n'
              'Either zip file: "' + filename + '" already exists!\n '
                                                'or..it was extracted already inside the current directory')
    if os.path.isfile(zipFilePath) and not os.path.isdir(fileDir):
        import zipfile
        print('Extracting the file....')
        zfileParent = zipFilePath.rstrip(filename)
        zfile = zipfile.ZipFile(zipFilePath, 'r')
        zfile.extractall(zfileParent)
        zfile.close()
        print('\t---> Extraction completed.')
    else:
        print('There is NO file to extract, or the destination directory exists already!')
