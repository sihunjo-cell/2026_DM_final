import os
import pymysql

# 1. MySQL/MariaDB 최고 관리자(root) 계정 정보 설정
#    비밀번호는 코드에 하드코딩하지 말고 환경변수 MYSQL_ROOT_PASSWORD 로 넘기세요.
#    예) PowerShell:  $env:MYSQL_ROOT_PASSWORD="본인root비번"; python before_run.py
#        bash:        MYSQL_ROOT_PASSWORD="본인root비번" python before_run.py
ROOT_USER = "root"
ROOT_PASSWORD = os.getenv("MYSQL_ROOT_PASSWORD", "")
HOST = "127.0.0.1"
PORT = 3306

if not ROOT_PASSWORD:
    raise SystemExit(
        "환경변수 MYSQL_ROOT_PASSWORD 가 비어 있습니다. "
        "본인 PC MySQL root 비밀번호를 환경변수로 설정한 뒤 다시 실행하세요."
    )

# 2. 생성할 데이터베이스 및 유저 정보 (chzzk-crawler/.env 의 DB_* 값과 동일해야 함)
DB_NAME = "chzzk_dm"
NEW_USER = "chzzk_user"
NEW_PASSWORD = "ChzzkCrawler2026!"

try:
    # 관리자 권한으로 데이터베이스 서버에 연결 (특정 DB 지정 안 함)
    conn = pymysql.connect(
        host=HOST,
        user=ROOT_USER,
        password=ROOT_PASSWORD,
        port=PORT,
        charset='utf8mb4'
    )
    cursor = conn.cursor()

    # 3. 데이터베이스 생성 (이미 존재하면 무시)
    print(f"[{DB_NAME}] 데이터베이스 생성을 시도합니다...")
    cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DB_NAME} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;")
    print("-> 데이터베이스 생성 완료.")

    # 4. 유저 생성 (이미 존재하면 무시)
    print(f"[{NEW_USER}] 유저 생성을 시도합니다...")
    cursor.execute(f"CREATE USER IF NOT EXISTS '{NEW_USER}'@'localhost' IDENTIFIED BY '{NEW_PASSWORD}';")
    
    # 만약 유저가 이미 존재했다면 비밀번호를 .env와 동일하게 강제 갱신
    cursor.execute(f"ALTER USER '{NEW_USER}'@'localhost' IDENTIFIED BY '{NEW_PASSWORD}';")
    print("-> 유저 생성 및 비밀번호 설정 완료.")

    # 5. 권한 부여
    print("권한을 부여합니다...")
    cursor.execute(f"GRANT ALL PRIVILEGES ON {DB_NAME}.* TO '{NEW_USER}'@'localhost';")
    
    # 6. 권한 적용
    cursor.execute("FLUSH PRIVILEGES;")
    print("-> 권한 부여 및 시스템 적용 완료.")

    print("\n모든 데이터베이스 초기 설정이 성공적으로 끝났습니다!")

except pymysql.err.OperationalError as e:
    print(f"\n데이터베이스 접속에 실패했습니다. root 비밀번호나 서버 실행 여부를 확인하세요.\n에러 내용: {e}")
except Exception as e:
    print(f"\n알 수 없는 오류가 발생했습니다:\n{e}")

finally:
    # 안전하게 연결 종료
    if 'conn' in locals() and conn.open:
        cursor.close()
        conn.close()