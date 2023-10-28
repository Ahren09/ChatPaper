import numpy as np
import os
import os.path as osp
import re
import datetime
import time
import openai, tenacity
import argparse
import configparser
import json
import tiktoken

import const
from get_paper import Paper
import jieba
from collections import namedtuple

ReviewerParams = namedtuple(
    "ReviewerParams",
    [
        "paper_path",
        "root",
        "file_format",
        "research_fields",
        "language",
        "task",
        "assignment_num"
    ],
)


def contains_chinese(text):
    for ch in text:
        if u'\u4e00' <= ch <= u'\u9fff':
            return True
    return False

def insert_sentence(text, sentence, interval):
    lines = text.split('\n')
    new_lines = []

    for line in lines:
        if contains_chinese(line):
            words = list(jieba.cut(line))
            separator = ''
        else:
            words = line.split()
            separator = ' '

        new_words = []
        count = 0

        for word in words:
            new_words.append(word)
            count += 1

            if count % interval == 0:
                new_words.append(sentence)

        new_lines.append(separator.join(new_words))

    return '\n'.join(new_lines)

# 定义Reviewer类
class Reviewer:
    # 初始化方法，设置属性
    def __init__(self, args=None):
        if args.language == 'en':
            self.language = 'English'
        elif args.language == 'zh':
            self.language = 'Chinese'
        else:
            self.language = 'Chinese'        
        # 创建一个ConfigParser对象
        self.config = configparser.ConfigParser()
        # 读取配置文件
        self.config.read('apikey.ini')
        # 获取某个键对应的值
        openai.api_base = self.config.get('AzureOPenAI', 'OPENAI_API_BASE')

        openai.api_version = self.config.get('AzureOPenAI', 'OPENAI_API_VERSION')
        openai.api_type = "azure"



        self.chat_api_list = self.config.get('AzureOPenAI', 'OPENAI_API_KEYS')[1:-1].replace('\'', '').split(',')
        self.chat_api_list = [api.strip() for api in self.chat_api_list if len(api) > 5]

        openai.api_key = self.chat_api_list[0]

        self.cur_api = 0
        self.file_format = args.file_format        
        self.max_token_num = 4096
        self.encoding = tiktoken.get_encoding("gpt2")

        self.paper_path = args.paper_path
    
    def validateTitle(self, title):
        # 修正论文的路径格式
        rstr = r"[\/\\\:\*\?\"\<\>\|]" # '/ \ : * ? " < > |'
        new_title = re.sub(rstr, "_", title) # 替换为下划线
        return new_title


    def grade_one_paper(self, paper, paper_index: int):
        messages = []

        prompt = f"You are a grader of a graduate-level machine learning class. Please grade a student's reading reflection based on a research paper he read. YOU MUST RETURN YOUR GRADING RESULTS USING THE TSV TABLE I PROVIDED. For `Reason to deduct point` in the table, leave it blank if no points are deducted"

        prompt += "\n\n" + "=" * 20 + const.HOMEWORK_GRADING_RUBRICS + "\n\n" + const.HOMEWORK_GRADING_TABLE

        text = "=" * 20 + paper.all_text

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": text},
        ]




        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            n=1,  # Number of responses you want to generate
            temperature=0.5,
            deployment_id="gpt35",
            # Controls the creativity of the generated response
        )
        result = response['choices'][0]["message"]["content"]


        print('='*30)
        print(result)

        usage = response['usage']

        return result





    def review_one_paper(self, paper, paper_index: int):
        htmls = []
        sections_of_interest = self.stage_1(paper)
        # extract the essential parts of the paper
        text = ''
        text += 'Title:' + paper.title + '. '
        text += 'Abstract: ' + paper.section_texts['Abstract']
        intro_title = next((item for item in paper.section_names if
                            'ntroduction' in item.lower()), None)
        if intro_title is not None:
            text += 'Introduction: ' + paper.section_texts[intro_title]
        # Similar for conclusion section
        conclusion_title = next(
            (item for item in paper.section_names if 'onclusion' in item), None)
        if conclusion_title is not None:
            text += 'Conclusion: ' + paper.section_texts[conclusion_title]
        for heading in sections_of_interest:
            if heading in paper.section_names:
                text += heading + ': ' + paper.section_texts[heading]
        chat_review_text = self.chat_review(text=text)
        htmls.append('## Paper:' + str(paper_index + 1))
        htmls.append('\n\n\n')
        htmls.append(chat_review_text)

        # 将审稿意见保存起来
        date_str = str(datetime.datetime.now())[:13].replace(' ', '-')
        try:
            export_path = os.path.join('./', 'outputs')
            os.makedirs(export_path)
        except:
            pass
        mode = 'w' if paper_index == 0 else 'a'
        file_name = os.path.join(export_path,
                                 date_str + '-' + self.validateTitle(
                                     paper.title) + "." + self.file_format)
        self.export_to_markdown("\n".join(htmls), file_name=file_name,
                                mode=mode)



    def review_by_chatgpt(self, paper_list):
        for paper_index, paper in enumerate(paper_list):
            self.review_one_paper(paper=paper, paper_index=paper_index)



    def stage_1(self, paper):

        text = ''
        text += 'Title: ' + paper.title + '. '

        if paper.section_texts.get('Abstract', ""):

            text += 'Abstract: ' + paper.section_texts.get('Abstract', "")


        text_token = len(self.encoding.encode(text))
        if text_token > self.max_token_num/2 - 800:
            input_text_index = int(len(text)*((self.max_token_num/2)-800)/text_token)
            text = text[:input_text_index]
        openai.api_key = self.chat_api_list[self.cur_api]
        self.cur_api += 1
        self.cur_api = 0 if self.cur_api >= len(self.chat_api_list)-1 else self.cur_api

        if reviewer_args.task == "review":

            prompt = f"You are a professional reviewer in the field of {reviewer_args.research_fields}. I will give you a paper. You need to review this paper and discuss the novelty and originality of ideas, correctness, clarity, the significance of results, potential impact and quality of the presentation. Due to the length limitations, I am only allowed to provide you the abstract, introduction, conclusion and at most two sections of this paper. Now I will give you the title and abstract and the headings of potential sections. You need to reply at most two headings. Then I will further provide you the full information, includes aforementioned sections and at most two sections you called for.\n\nTitle: {paper.title}\n\n Abstract: {paper.section_texts['Abstract']}\n\n Potential Sections: {paper.section_names[2:-1]}\n\nFollow the following format to output your choice of sections:{{chosen section 1}}, {{chosen section 2}}\n\n"

        elif reviewer_args.task == "scholarship":

            prompt = f"You are a Ph.D. student in the field of {reviewer_args.research_fields}. I will give you my first-authored paper. You need to review this paper and write a  scholarship application centered around how my research contributes to societal progress based on my paper. " \
                     f"Due to the length limitations, I am only allowed to provide you the abstract, introduction, conclusion and at most two sections of this paper. Now I will give you the title and abstract and the headings of potential sections. You need to reply at most two headings. Then I will further provide you the full information, includes aforementioned sections and at most two sections you called for.\n\n" \
                     f"Title: {paper.title}\n\n" \
                     f"Abstract: {paper.section_texts['Abstract']}\n\n" \
                     f"Potential Sections: {paper.section_names[2:-1]}\n\n" \
                     f"Follow the following format to output your choice of sections:{{chosen section 1}}, {{chosen section 2}}\n\n"

        messages = [
            {"role": "system",
             "content": prompt},
            {"role": "user", "content": text},
        ]
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            deployment_id="gpt35",
            # api_version=openai.__version__,
        )
        result = ''
        for choice in response.choices:
            result += choice.message.content
        print(result)
        return result.split(',')




    @tenacity.retry(wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
                    stop=tenacity.stop_after_attempt(5),
                    reraise=True)
    def chat_review(self, text):
        openai.api_key = self.chat_api_list[self.cur_api]
        self.cur_api += 1
        self.cur_api = 0 if self.cur_api >= len(self.chat_api_list)-1 else self.cur_api
        review_prompt_token = 1000
        text_token = len(self.encoding.encode(text))
        input_text_index = int(len(text)*(self.max_token_num-review_prompt_token)/text_token)
        input_text = "This is the paper for your review:" + text[:input_text_index]
        with open('ChatReviewerAndResponse/ReviewFormat_AAAI.txt', 'r') as file:   # 读取特定的审稿格式
            review_format = file.read()


        HOMEWORK_GRADING_FORMAT = """
        ## Format
        
        Problem description\tX points
        Importance and why should we care?\tX points
        Describe the proposed method.\tX points
        Describe the main experimental results.\tX points
        Strengths\tX points
        Weaknesses\tX points
        
        """


        if reviewer_args.task == "review":
            content = "You are a professional reviewer in the field of "+reviewer_args.research_fields+". Now I will give you a paper. You need to give a complete review opinion according to the following requirements and format:"+ review_format +" Please answer in {}.".format(self.language)

        elif reviewer_args.task == "grading":
            content = f"You are a grader of a research class related to "+reviewer_args.research_fields+". Now I will give you a student's report based on a research paper he read. Please grade his report according to the following grading rubrics and format:\n\n{HOMEWORK_GRADING_RUBRICS}\n\nFormat\n\n."

        elif reviewer_args.task == "scholarship":
            content = "You are a Ph.D. student in the field of "+reviewer_args.research_fields+". Now I will give you my first-authored paper. You need to write a scholarship application centered around how my research contributes to societal progress based on my paper. Please answer in {}.".format(self.language)

        else:
            raise ValueError("Invalid task type: {}".format(reviewer_args.task))

        messages = [
            {"role": "system", "content": content},
            {"role": "user", "content": input_text},
        ]
                
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            deployment_id="gpt35",
        )
        result = ''
        for choice in response.choices:
            result += choice.message.content

        print("********"*10)
        print(result)
        print("********"*10)
        print("prompt_token_used:", response.usage.prompt_tokens)
        print("completion_token_used:", response.usage.completion_tokens)
        print("total_token_used:", response.usage.total_tokens)
        print("response_time:", response.response_ms/1000.0, 's')                    
        return result        
                        
    def export_to_markdown(self, text, file_name, mode='w'):
        # 使用markdown模块的convert方法，将文本转换为html格式
        # html = markdown.markdown(text)
        # 打开一个文件，以写入模式
        with open(file_name, mode, encoding="utf-8") as f:
            # 将html格式的内容写入文件
            f.write(text)                    

def chat_reviewer_main(args):            

    reviewer1 = Reviewer(args=args)
    # 开始判断是路径还是文件：   
    paper_list = []
    print(os.listdir(args.paper_path))

    if args.paper_path.endswith(".pdf"):
        paper_list.append(Paper(path=args.paper_path))            
    else:
        idx_paper = 0


        path_grades = osp.join(args.paper_path, f'grades_HW{reviewer_args.assignment_num}.json')

        if osp.exists('grading.json'):
            results_d = json.load(open('grading.json', 'r', encoding='utf-8'))

        else:
            results_d = {}

        for filename in os.listdir(args.paper_path):
            if reviewer_args.task == "grading" and  filename in results_d:
                continue

            print("=" * 20)
            # 如果找到PDF文件，则将其复制到目标文件夹中
            if filename.endswith(".pdf") and filename == "10654.pdf" or filename.endswith(".txt") or filename.endswith(".md"):
                # paper_list.append(Paper(path=os.path.join(root, filename)))

                if reviewer_args.task == "grading":
                    paper = Paper(path=os.path.join(args.paper_path, filename), grading=True)
                    result = reviewer1.grade_one_paper(paper, idx_paper)

                    results_d[filename] = result

                    json.dump(results_d,
                              open(f'grades_HW{reviewer_args.assignment_num}.json', 'w', encoding='utf-8'),
                              ensure_ascii=False, indent=2)

                else:
                    paper = Paper(path=os.path.join(args.paper_path, filename))
                    reviewer1.review_one_paper(paper, idx_paper)
                idx_paper += 1



    # print(f"# Papers: {len(paper_list)}")

    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--root", type=str, default='/Users/ahren/Library/CloudStorage/OneDrive-GeorgiaInstituteofTechnology/Courses/CSE8803DSN/Grading', help="path of papers")
    parser.add_argument("--paper_path", type=str, default='/Users/ahren/Library/CloudStorage/OneDrive-GeorgiaInstituteofTechnology/PaperReview/AAAI2024', help="path of papers")
    parser.add_argument("--file_format", type=str, default='pdf', help="output file format")
    parser.add_argument("--research_fields", type=str, default='computer science, artificial intelligence and reinforcement learning, misinformation, social networks', help="the research fields of paper")
    parser.add_argument("--language", type=str, default='en', help="output lauguage, en or zh")
    parser.add_argument("--task", type=str, choices=["review", "scholarship", "grading"], default='review', help="")
    parser.add_argument("--assignment_num", type=int, default=-1, help="")

    args = parser.parse_args()
    if args.task == "grading":
        args.paper_path = f"/Users/ahren/Library/CloudStorage/OneDrive-GeorgiaInstituteofTechnology/Courses/CSE8803DSN/Grading/Assignment{args.assignment_num}/"


    reviewer_args = ReviewerParams(**vars(args))



    start_time = time.time()
    chat_reviewer_main(args=reviewer_args)
    print("review time:", time.time() - start_time)
