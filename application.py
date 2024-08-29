import logging
import uuid
import requests
from bs4 import BeautifulSoup
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_openai import ChatOpenAI
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_teddynote.prompts import load_prompt
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnableMap
import fitz  # PyMuPDF
import pandas as pd
import re
import os
from dotenv import load_dotenv
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import Header  # Header 모듈 임포트

# 환경 변수 로드
load_dotenv()

application = Flask(__name__)
CORS(application)

# 204 응답에 대한 로그를 비활성화하기 위한 로거 설정
log = logging.getLogger("werkzeug")
log.setLevel(logging.WARNING)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("my_logger")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

conversation_states = {}


@application.before_request
def before_request():
    if request.path.startswith("/socket.io"):
        return "", 204


def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text


def extract_text_from_excel(file):
    df = pd.read_excel(file)
    return df.to_string(index=False)


def extract_retirement_clause(text):
    # 퇴직금 관련 조항을 정규표현식을 통해 추출
    pattern = re.compile(r"퇴직[금|급여].*?(?:\n|$)", re.DOTALL)
    matches = pattern.findall(text)
    return "\n".join(matches) if matches else ""


def analyze_retirement_clause(retirement_clause, re_summary_input=""):
    if not retirement_clause:
        return "정관에 임원 퇴직급여 규정이 필요합니다."

    llm = ChatOpenAI(
        model_name="gpt-4o", temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    output_parser = StrOutputParser()

    prompt_messages = [
        (
            "system",
            "당신은 법률 전문가입니다. 아래 텍스트는 정관 파일에서 임원 퇴직금 관련한 텍스트를 추출한 것이다. 임원 퇴직급여에 대한 구체적인 지급 규정이 포함되어 있는지 판단해 주세요. 오로지 정관에만 임원 퇴직 급여에 대한 구체적 규정이 있어야 한다. 정관에 없다면 잘못된 것이다.",
        ),
        ("user", f"퇴직금 관련 조항: {retirement_clause}"),
    ]

    if re_summary_input:
        prompt_messages.append(("user", f"추가 요청 사항: {re_summary_input}"))

    prompt = ChatPromptTemplate.from_messages(prompt_messages)

    chain = prompt | llm | output_parser
    response = chain.invoke({})

    return response


def generate_consulting_email(analysis_result, retirement_payment):
    formatted_payment = (
        f"{retirement_payment:,.0f}"  # 소수점 제거하고 천 단위로 쉼표 추가
    )
    llm = ChatOpenAI(
        model_name="gpt-4o", temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    output_parser = StrOutputParser()

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "당신은 세무 이메일 작성 전문가입니다. 다음 분석 결과와 퇴직금 계산 결과를 바탕으로 고객에게 보낼 컨설팅 이메일을 반드시 한국어로 작성해 주세요. 양식은 다음과 같습니다. 1. 현재 정관 상태, 2. 정관에서의 퇴직금 규정의 중요성, 3. 예상 퇴직금 규모, 4. 정관컨설팅의 필요성 및 기대효과, 5. 컨설팅 추천",
            ),
            (
                "user",
                f"분석 결과: {analysis_result}\n퇴직금 계산 결과: {formatted_payment}만원",
            ),
        ]
    )

    chain = prompt | llm | output_parser
    response = chain.invoke({})

    return response


def crawl_article(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    article_body = soup.find("div", {"itemprop": "articleBody"})

    if article_body:
        article_text = ""
        for element in article_body.stripped_strings:
            article_text += element + "\n"

        # 공백 줄 제거
        article_text = "\n".join(
            [line.strip() for line in article_text.splitlines() if line.strip()]
        )
        return article_text
    else:
        return "기사 본문을 찾을 수 없습니다."


def summarize_article(article_text, reSummaryInput):
    llm = ChatOpenAI(
        model_name="gpt-4o", temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    output_parser = StrOutputParser()

    prompt_messages = [
        (
            "system",
            "당신은 기사 요약 전문가입니다. 다음 기사를 상세하게 요약하고 향후 미칠 효과에 대하여 추론하세요. 추가 요청 사항이 있다면 해당 사항을 중점적으로 요약하고 추론하세요",
        ),
        ("user", f"기사 내용: {article_text}"),
    ]

    if reSummaryInput:
        prompt_messages.append(("user", f"추가 요청 사항: {reSummaryInput}"))

    prompt = ChatPromptTemplate.from_messages(prompt_messages)

    chain = prompt | llm | output_parser
    response = chain.invoke({})

    return response


def create_email_content(summary):
    llm = ChatOpenAI(
        model_name="gpt-4o", temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    output_parser = StrOutputParser()

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "당신은 이메일 작성 전문가입니다. 다음 뉴스 기사 요약을 토대로 고객에게 보낼 이메일을 한국어로 작성하세요.",
            ),
            ("user", f"기사 요약: {summary}"),
        ]
    )

    chain = prompt | llm | output_parser
    response = chain.invoke({})

    return response


@application.route("/api/summarize", methods=["POST"])
def summarize():
    data = request.json
    url = data.get("url")
    reSummaryInput = data.get("reSummaryInput")

    if not url:
        return jsonify({"error": "URL is required"}), 400

    try:
        article_text = crawl_article(url)
        summary = summarize_article(article_text, reSummaryInput)
        email_content = create_email_content(summary)
        return jsonify({"summary": email_content})
    except Exception as e:
        logger.error(f"Error summarizing article: {e}")
        return jsonify({"error": str(e)}), 500


@application.route("/api/test", methods=["POST"])
def test():
    data = request.json
    conversation_id = data.get("conversation_id")
    topic = data.get("topic")
    tone = data.get("tone", "부드러운")  # 기본 어조 설정

    # 어조 설정에 따른 설명
    if tone == "상냥한":
        tone_description = "20대 여사원이 40-50대 대표들에게 상냥하게 응대하는 모드"
    elif tone == "전문적인":
        tone_description = "세무법인 직원으로 전문적인 어조로 응대하는 모드"
    else:
        tone_description = ""

    # 대화 히스토리 관리
    if conversation_id in conversation_states:
        history = conversation_states[conversation_id]
    else:
        history = []

    # 새로운 메시지 추가
    history.append(
        SystemMessage(
            content="당신은 친절한 리치디바인 세무사 어시스턴트입니다. 고객 질문에 대하여 어조에 알맞게 답변을 하십시오."
        )
    )
    history.append(HumanMessage(content=f"어조: {tone_description}"))
    history.append(HumanMessage(content=f"고객 질문: {topic}"))

    conversation_states[conversation_id] = history[-12:]

    # LLM 호출 및 응답 생성
    llm = ChatOpenAI(
        model_name="gpt-4o",
        temperature=0,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )

    prompt = ChatPromptTemplate.from_messages(conversation_states[conversation_id])
    response = llm.invoke(prompt.format_messages())

    # 응답 메시지를 히스토리에 추가
    ai_message = AIMessage(content=response.content)
    history.append(ai_message)

    # 결과 반환
    return jsonify({"answer": ai_message.content, "conversation_id": conversation_id})


@application.route("/api/ask", methods=["POST"])
def ask():
    data = request.json
    conversation_id = data.get("conversation_id")
    topic = data.get("topic")
    keywords = data.get("keywords")
    product_info = data.get("product_info")
    product_details = data.get("product_details")
    perspective = data.get("perspective")
    tone_and_manner = data.get("tone_and_manner")
    notes = data.get("notes")
    selected_title = data.get("selected_title", "")
    message = data.get("message", "")
    content = data.get("content", "")
    prompt_type = data.get("prompt_type")
    previous_messages = data.get("previous_messages", [])
    tone = data.get("tone", "부드러운")  # 기본 어조 설정

    if not conversation_id:
        conversation_id = str(uuid.uuid4())

    if not topic or not prompt_type:
        return jsonify({"error": "Invalid input"}), 400

    if tone == "상냥한":
        tone_description = "20대 여사원이 40-50대 대표들에게 상냥하게 응대하는 모드"
    elif tone == "전문적인":
        tone_description = "세무법인 직원으로 전문적인 어조로 응대하는 모드"
    else:
        tone_description = ""

    prompt_messages = [
        ("system", f"{tone_description}"),
    ]

    for msg in previous_messages:
        role = "user" if msg["isUser"] else "assistant"
        prompt_messages.append((role, msg["text"]))

    prompt_messages.append(("user", f"대화 ID: {conversation_id}"))
    prompt_messages.append(("user", f"주제: {topic}"))

    if prompt_type == "기본모드":
        prompt_messages.extend(
            [
                (
                    "system",
                    "당신은 친절한 세무사 어시스턴트입니다. 고객 질문에 대하여 어조에 알맞게 답변을 하십시오.",
                ),
                ("user", f"어조: {tone_description}"),
                ("user", f"고객 질문: {topic}"),
                ("assistant", "세무사 답변:"),
            ]
        )
    elif prompt_type == "세법기사":
        prompt_messages.extend(
            [
                (
                    "system",
                    "당신은 세법 전문가입니다. 다음 텍스트를 요약하고, 그 요약을 바탕으로 세법 기사가 미칠 향후 영향을 설명해 주세요.",
                ),
                ("user", f"세법 기사: {content}"),
            ]
        )
    elif prompt_type == "블로그":
        prompt = load_prompt("prompts/sns.yaml", encoding="utf-8")
    elif prompt_type == "엑셀":
        prompt_messages.extend(
            [
                (
                    "system",
                    "당신은 친절한 AI 세무사 어시스턴트입니다. 다음 엑셀 파일의 정보를 기반으로 질문에 답변하세요.",
                ),
                ("user", f"엑셀 파일 정보: {product_info}"),
                ("user", f"고객 질문: {topic}"),
                ("assistant", "세무사 답변:"),
            ]
        )
    elif prompt_type == "회사정보":
        prompt_messages.extend(
            [
                (
                    "system",
                    "당신은 친절한 AI 세무사 어시스턴트입니다. 다음 회사 정보를 기반으로 질문에 답변하세요.",
                ),
                ("user", f"회사 정보: {product_info}"),
                ("user", f"고객 질문: {topic}"),
                ("assistant", "세무사 답변:"),
            ]
        )
    elif prompt_type == "제목":
        prompt_messages.extend(
            [
                ("system", "다음 정보를 바탕으로 3개의 제목을 작성해 주세요."),
                (
                    "user",
                    f"주제: {topic}, 키워드: {keywords}, 제품정보: {product_info}, 제품 상세: {product_details}, 관점: {perspective}, 톤앤 매너 선택: {tone_and_manner}, 참고사항: {notes}",
                ),
            ]
        )
    elif prompt_type == "수정":
        prompt_messages.extend(
            [
                (
                    "system",
                    "당신은 친절한 AI 블로그 어시스턴트입니다. 다음 정보를 기반으로 기존 본문을 수정해 주세요.",
                ),
                ("user", f"기존 본문: {content}"),
                ("user", f"수정 요청: {message}"),
                ("assistant", "수정된 본문:"),
            ]
        )

    prompt = ChatPromptTemplate.from_messages(prompt_messages)

    llm = ChatOpenAI(
        model_name="gpt-4o", temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    output_parser = StrOutputParser()

    chain = prompt | llm | output_parser
    response = chain.invoke(
        {
            "topic": topic,
            "keywords": keywords,
            "product_info": product_info,
            "product_details": product_details,
            "perspective": perspective,
            "tone_and_manner": tone_and_manner,
            "notes": notes,
            "selected_title": selected_title,
            "message": message,
            "content": content,
        }
    )
    ai_answer = "".join([token for token in response])

    if prompt_type == "제목":
        titles = ai_answer.split("\n")
        return jsonify({"titles": titles, "conversation_id": conversation_id})
    else:
        return jsonify({"answer": ai_answer, "conversation_id": conversation_id})


@application.route("/api/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    monthly_salary = request.form.get("monthly_salary", type=float)
    days_of_service = request.form.get("days_of_service", type=float)
    re_summary_input = request.form.get("re_summary_input", "")  # 추가: 재분석 입력값

    if file.filename.endswith(".pdf"):
        company_info = extract_text_from_pdf(file)
    elif file.filename.endswith(".xlsx"):
        company_info = extract_text_from_excel(file)
    else:
        return jsonify({"error": "Unsupported file type"}), 400

    retirement_clause = extract_retirement_clause(company_info)
    analysis_result = analyze_retirement_clause(retirement_clause, re_summary_input)
    retirement_payment = calculate_retirement_payment(monthly_salary, days_of_service)

    email_template = generate_consulting_email(analysis_result, retirement_payment)

    return jsonify(
        {
            "company_info": company_info,
            "retirement_clause": retirement_clause,
            "analysis_result": analysis_result,
            "retirement_payment": retirement_payment,
            "email_template": email_template,
        }
    )


def calculate_retirement_payment(monthly_salary, days_of_service):
    # 퇴직금 계산
    retirement_payment = monthly_salary * (days_of_service / 365)

    return retirement_payment


@application.route("/api/send-email", methods=["POST"])
def send_email():
    data = request.json
    to_email = data.get("to")
    subject = data.get("subject")
    text = data.get("text")

    if not to_email or not subject or not text:
        return jsonify({"error": "Invalid input"}), 400

    # SMTP 설정
    smtp_server = os.getenv("SMTP_SERVER")
    smtp_port = int(os.getenv("SMTP_PORT"))
    smtp_user = os.getenv("SMTP_USER")
    smtp_password = os.getenv("SMTP_PASSWORD")

    logger.info(f"Sending email to {to_email} with subject '{subject}'")

    # 이메일 구성
    msg = MIMEMultipart()
    msg["From"] = smtp_user
    msg["To"] = Header(to_email, "utf-8")
    msg["Subject"] = Header(subject, "utf-8")

    # 본문 추가
    body = MIMEText(text, "plain", "utf-8")
    msg.attach(body)

    try:
        logger.info("Connecting to the SMTP server...")
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.set_debuglevel(1)  # 디버깅 출력을 활성화합니다.
        server.ehlo("localhost")  # ASCII 문자만 포함된 호스트 이름을 사용합니다.
        server.starttls()
        server.ehlo("localhost")
        logger.info("Logged in to the SMTP server")
        server.login(smtp_user, smtp_password)
        logger.info("Sending email...")
        server.sendmail(smtp_user, to_email, msg.as_string().encode("utf-8"))
        server.quit()
        logger.info("Email sent successfully")
        return jsonify({"message": "Email sent successfully"}), 200
    except smtplib.SMTPException as smtp_error:
        logger.error(f"SMTP error: {smtp_error}")
        return jsonify({"error": "Failed to send email"}), 500
    except Exception as e:
        logger.error(f"Error sending email: {e}")
        return jsonify({"error": "Failed to send email"}), 500


# GPT-4를 사용하여 응답 생성
def generate_gpt_response(comparison_report, re_summary_input=None):
    llm = ChatOpenAI(
        model_name="gpt-4o", temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    output_parser = StrOutputParser()

    # Prepare prompt messages
    prompt_messages = [
        (
            "system",
            "당신은 미수금 관리 전문가입니다. 다음 데이터를 바탕으로 미수금 독촉이 필요한 회사를 판단하고, 회사명, 이메일과 독촉이 필요한 이유를 '회사명 : 이메일 : 이유' 형식으로 명확히 나열하세요.",
        ),
        ("user", f"미수금 데이터: {comparison_report}"),
    ]

    if re_summary_input:
        prompt_messages.append(("user", f"추가 요청 사항: {re_summary_input}"))

    prompt = ChatPromptTemplate.from_messages(prompt_messages)
    chain = prompt | llm | output_parser
    response = chain.invoke({})

    # 결과를 파싱해서 JSON 형식으로 변환
    lines = response.split("\n")
    results = []
    for line in lines:
        if ": " in line:
            parts = line.split(": ")
            if len(parts) == 3:
                company, email, reason = parts
                results.append(
                    {
                        "company": company.strip(),
                        "email": email.strip(),
                        "reason": reason.strip(),
                    }
                )

    return results


# 엑셀 데이터를 비교하는 함수
def compare_excel_files(prev_df, curr_df):
    report = []

    for company in prev_df.index:
        if company in curr_df.index:
            prev_row = prev_df.loc[company]
            curr_row = curr_df.loc[company]

            prev_balance = prev_row["잔액"]
            curr_balance = curr_row["잔액"]
            prev_debit = prev_row["차변"]
            curr_debit = curr_row["차변"]
            prev_credit = prev_row["대변"]
            curr_credit = curr_row["대변"]
            email = curr_row["이메일"]

            if (
                curr_balance == prev_balance
                and curr_debit == prev_debit
                and curr_credit == prev_credit
            ):
                report.append(
                    f"{company}: {email}: 잔액, 차변, 대변 모두 변동이 없으므로 독촉이 필요하지 않습니다."
                )
            elif curr_debit > prev_debit and curr_credit > prev_credit:
                report.append(
                    f"{company}: {email}: 매출 발생과 변제가 동시에 이루어졌으므로 독촉이 필요하지 않습니다."
                )
            elif curr_balance == prev_balance:
                report.append(
                    f"{company}: {email}: 잔액이 변동하지 않았으므로 독촉이 필요합니다."
                )
            elif curr_debit > prev_debit and curr_credit == prev_credit:
                report.append(
                    f"{company}: {email}: 매출이 증가했으나 변제가 이루어지지 않았으므로 독촉이 필요합니다."
                )
            else:
                report.append(
                    f"{company}: {email}: 거래가 중단된 상태이거나 변제가 이루어지지 않았으므로 독촉이 필요합니다."
                )
        else:
            email = prev_df.loc[company, "이메일"]
            report.append(
                f"{company}: {email}: 현재 데이터에서 찾을 수 없습니다. 거래가 중단되었을 수 있습니다."
            )

    return "\n".join(report)


# 엑셀 파일 비교 후 GPT-4를 통해 설명 생성
@application.route("/api/compare", methods=["POST"])
def compare():
    try:
        prev_file = request.files["previousMonthFile"]
        curr_file = request.files["currentMonthFile"]

        prev_df = pd.read_excel(prev_file, index_col="거래처")
        curr_df = pd.read_excel(curr_file, index_col="거래처")

        comparison_report = compare_excel_files(prev_df, curr_df)
        gpt_response = generate_gpt_response(comparison_report)

        return jsonify({"report": comparison_report, "gpt_response": gpt_response})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    application.run(debug=True)
