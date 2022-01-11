#!/usr/bin/env python3
import datetime
import argparse
import hashlib
import imghdr
import os
import pickle
import posixpath
import re
import signal
import socket
import threading
import time
import urllib.parse
import urllib.request

socket.setdefaulttimeout(3)
urlopenheader = {'User-Agent': 'Mozilla/5.0 (X11; Fedora; Linux x86_64; rv:94.0) Gecko/20100101 Firefox/94.0'}


def checker(output_dir, limit):
    while True:
        time.sleep(0.1)
        file_list = os.listdir(output_dir)
        if len(file_list) >= limit:
            raise RuntimeError


class dl:
    def __init__(self, base_root, keyword: str, limit: int = 30):
        self.in_progress = 0
        self.limit = limit
        self.pool_sema = threading.BoundedSemaphore(20)
        self.keyword = keyword
        self.output_dir = os.path.join(base_root, keyword)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        file_list = os.listdir(self.output_dir)
        if len(file_list) >= self.limit:
            print("{}的图片已经够啦".format(self.keyword))
            raise RuntimeError
        self.img_sema = threading.Semaphore()

        self.tried_urls = []
        self.image_md5s = {}
        self.done = False
        self.checker_break = 0

    def download(self, url: str):

        if url in self.tried_urls:
            print('SKIP: Already checked url, skipping')
            return
        self.pool_sema.acquire()
        self.in_progress += 1
        acquired_img_sema = False
        path = urllib.parse.urlsplit(url).path
        filename = posixpath.basename(path).split('?')[0]  # Strip GET parameters from filename
        name, ext = os.path.splitext(filename)
        name = name[:36].strip()
        filename = name + ext

        try:
            request = urllib.request.Request(url, None, urlopenheader)
            image = urllib.request.urlopen(request).read()
            if not imghdr.what(None, image) or self.done:
                if self.done:
                    pass
                else:
                    print('SKIP: Invalid image, not saving ' + filename)
                return

            md5_key = hashlib.md5(image).hexdigest()
            if md5_key in self.image_md5s or self.done:
                if self.done:
                    pass
                else:
                    print('SKIP: Image is a duplicate of ' + self.image_md5s[md5_key] + ', not saving ' + filename)
                return

            i = 0
            # while os.path.exists(os.path.join(self.output_dir, filename)):
            #     if hashlib.md5(open(os.path.join(self.output_dir, filename), 'rb').read()).hexdigest() == md5_key or self.done:
            #         print('SKIP: Already downloaded ' + filename + ', not saving')
            #         return
            #     i += 1
            #     filename = "%s-%d%s" % (name, i, ext)

            self.image_md5s[md5_key] = filename

            self.img_sema.acquire()
            acquired_img_sema = True
            if self.limit is not None and len(self.tried_urls) >= self.limit or self.done:
                return
            filename = "{}_{}.jpg".format(self.keyword,len(self.tried_urls))
            imagefile = open(os.path.join(self.output_dir, filename), 'wb')
            imagefile.write(image)
            imagefile.close()
            print(" OK : " + filename)
            self.tried_urls.append(url)
        except Exception as e:
            print("FAIL: " + filename)
        finally:
            self.pool_sema.release()
            if acquired_img_sema:
                self.img_sema.release()
            self.in_progress -= 1

    def fetch_photo(self):
        print("==" * 30)
        time1_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print("开始爬取{}的图片! --{} ".format(self.keyword,  time1_str))
        print("==" * 30)

        current = 0
        last = ''
        t_check = threading.Thread(target=self.checker)
        t_check.start()
        while True:
            time.sleep(0.1)

            if self.in_progress > 10 or self.done:
                if self.done:
                    break
                continue

            request_url = 'https://www.bing.com/images/async?q=' + urllib.parse.quote_plus(
                self.keyword) + '&first=' + str(
                current) + '&count=35&qft='
            request = urllib.request.Request(request_url, None, headers=urlopenheader)
            response = urllib.request.urlopen(request)
            html = response.read().decode('utf8')
            links = re.findall('murl&quot;:&quot;(.*?)&quot;', html)
            try:
                if links[-1] == last:
                    return
                for index, link in enumerate(links):
                    if self.limit is not None and len(self.tried_urls) >= self.limit:
                        exit(0)
                    t = threading.Thread(target=self.download, args=(link,))
                    t.start()
                    current += 1
                last = links[-1]
            except IndexError:
                print('FAIL: No search results for "{0}"'.format(self.keyword))
                return


    def checker(self):
        """
        辅助线程, 检查过程是否已完成
        :return:
        """
        sleep_time = 0.1
        while True:
            if self.checker_break==3:break
            time.sleep(sleep_time)
            file_list = os.listdir(self.output_dir)
            if len(file_list) >= self.limit:
                self.done = True
                print("==" * 30)
                time1_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                print("{}数量已完成, 正在等待子线程结束{} --{} ".format(self.keyword,self.done, time1_str))
                print("==" * 30)
                sleep_time = 1
                self.checker_break+=1


def scraper(names:list,base_root:str,limit:int=30):
    for name in names:
        try:
            DL = dl(base_root, name, limit=limit)
            file_list = os.listdir(DL.output_dir)
            if len(file_list) >= DL.limit:
                print("{}的图片已经够啦".format(DL.keyword))
                raise RuntimeError
            DL.fetch_photo()
            del DL
        except :
            print("{} 已下载够了".format(name))

