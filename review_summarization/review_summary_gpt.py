import os
import openai
import chardet #파일 인코딩 감지


# OpenAI API 키
openai.api_key = "api 키를 입력하세요"

def summarize_and_save_directory(directory, max_length=3000):
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read().replace("\t", "|")
                # 텍스트의 길이를 최대 토큰 수에 맞춰 조정
                if len(text) > max_length:
                    text = text[:max_length]
                summary = get_chat_gpt_summary(text)
                save_summary(summary, file_path)

def get_chat_gpt_summary(text):
    directive = """
    - This is a chatbot that summarizes the reviews of companies.
    - The result should be in Korean.
    - Need an overall summary of the tone of the company, not a specific company.
    - Don't summarize about salary, and atmosphere and the balance between work and life.
    - summarize in terms of benefits of company.
    """
    session = [{"role": "system", "content": directive}]
    message = {"role": "user", "content": f"Summarize the review in a sentence: {text}"}
    session.append(message)
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=session
    )

    output_text = response["choices"][0]["message"]["content"]
    session.append({"role": "system", "content": output_text})
    
    return output_text

def save_summary(summary, original_file_path):
    summary_filename = original_file_path.replace(".txt", "_summary.txt")
    with open(summary_filename, 'w', encoding='utf-8') as file:
        file.write(summary)
        print(f"요약본이 저장된 파일: {summary_filename}")

# 디렉토리 지정
directory_path = "C:/Users/Park/Desktop/2"

# 디렉토리 내의 파일들 요약 및 저장
summarize_and_save_directory(directory_path)


def convert_encoding(file_path, original_encoding, target_encoding):
    with open(file_path, 'r', encoding=original_encoding, errors='ignore') as file:
        text = file.read()

    with open(file_path, 'w', encoding=target_encoding) as file:
        file.write(text)

    print(f"File {file_path} has been converted to {target_encoding} encoding.")

#파일위치, 바꾸는 인코딩
file_path = 'C:/Users/Park/Desktop/1/pos_balance.txt'
convert_encoding(file_path, 'CP949', 'utf-8')


# 파일 경로
file_path = 'C:/Users/Park/Desktop/1/pos_balance.txt'

def detect_file_encoding(file_path):
    with open(file_path, 'rb') as file:
        raw_data = file.read(5000)
        result = chardet.detect(raw_data)
        return result

encoding_info = detect_file_encoding(file_path)
print(encoding_info)