import random
import sys
from datetime import date, timedelta
import pymysql

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Cấu hình kết nối MySQL Laragon
DB_CONFIG = {
    "host": "localhost",
    "port": 3306,
    "user": "HUY0406",  # Hoặc 'root'
    "password": "",      # Mật khẩu trống của Laragon
    "database": "hospital",
    "charset": "utf8mb4",
    "autocommit": True,
}


def seed_mysql_data():
    print("⏳ Đang kết nối vào MySQL để nạp dữ liệu mẫu...")
    conn = pymysql.connect(**DB_CONFIG)
    cursor = conn.cursor()

    try:
        # Xóa dữ liệu cũ nếu có
        cursor.execute("SET FOREIGN_KEY_CHECKS = 0;")
        cursor.execute("TRUNCATE TABLE appointments;")
        cursor.execute("TRUNCATE TABLE schedules;")
        cursor.execute("TRUNCATE TABLE doctors;")
        cursor.execute("TRUNCATE TABLE departments;")
        cursor.execute("TRUNCATE TABLE patients;")
        cursor.execute("SET FOREIGN_KEY_CHECKS = 1;")

        # 1. Thêm 6 Chuyên khoa (bao gồm Khoa Khám bệnh tổng quát)
        departments = [
            ("KHAM_TONG_QUAT", "Khoa Khám bệnh tổng quát", "Khám nội tổng quát, tầm soát sức khỏe định kỳ, sốt chưa rõ nguyên nhân, sốt xuất huyết, mệt mỏi suy nhược."),
            ("TIM_MACH", "Khoa Tim mạch", "Khám và điều trị tăng huyết áp, rối loạn nhịp tim, suy tim, bệnh mạch vành."),
            ("THAN_KINH", "Khoa Thần kinh", "Chuyên trị đau đầu, chóng mặt, rối loạn tiền đình, mất ngủ, đau dây thần kinh."),
            ("TAI_MUI_HONG", "Khoa Tai Mũi Họng", "Điều trị viêm xoang, viêm họng, viêm amidan, ù tai, nghẹt mũi, khàn tiếng."),
            ("TIEU_HOA", "Khoa Tiêu hóa & Gan mật", "Khám chữa đau dạ dày, trào ngược thực quản, viêm đại tràng, men gan cao."),
            ("HO_HAP", "Khoa Hô hấp & Phổi", "Chẩn đoán và điều trị viêm phế quản, hen suyễn, COPD, ho kéo dài, khó thở."),
        ]
        sql_dept = "INSERT INTO departments (code, name, description) VALUES (%s, %s, %s);"
        cursor.executemany(sql_dept, departments)
        print("✅ Đã nạp 6 Chuyên khoa.")

        # Lấy ID của các khoa vừa tạo
        cursor.execute("SELECT code, id FROM departments;")
        dept_map = dict(cursor.fetchall())

        # 2. Thêm 12 Bác sĩ chuyên khoa
        doctors = [
            # Khoa Khám bệnh tổng quát
            (dept_map["KHAM_TONG_QUAT"], "BS.CKII Nguyễn Văn Hùng", "Bác sĩ Chuyên khoa II", "Phòng 001 - Khu Khám bệnh", "Hơn 20 năm kinh nghiệm nội tổng quát, tầm soát bệnh lý đa cơ quan, sốt nhiễm trùng và bệnh truyền nhiễm."),
            (dept_map["KHAM_TONG_QUAT"], "ThS.BS Lê Thị Lan", "Thạc sĩ Bác sĩ", "Phòng 002 - Khu Khám bệnh", "Chuyên khám và điều trị bệnh nội khoa tổng hợp, kiểm tra sức khỏe tổng quát định kỳ."),
            # Khoa Tim mạch
            (dept_map["TIM_MACH"], "PGS.TS.BS Trần Văn An", "Phó Giáo sư, Tiến sĩ", "Phòng 101 - Khu A", "Hơn 25 năm kinh nghiệm về can thiệp tim mạch, điều trị tăng huyết áp và suy tim."),
            (dept_map["TIM_MACH"], "BS.CKII Nguyễn Thị Mai", "Bác sĩ Chuyên khoa II", "Phòng 102 - Khu A", "Chuyên gia siêu âm tim Doppler màu, chẩn đoán bệnh mạch vành và van tim."),
            # Khoa Thần kinh
            (dept_map["THAN_KINH"], "TS.BS Lê Hoàng Nam", "Tiến sĩ Bác sĩ", "Phòng 201 - Khu B", "Chuyên gia đầu ngành về rối loạn tiền đình, đau nửa đầu Migraine và suy nhược thần kinh."),
            (dept_map["THAN_KINH"], "ThS.BS Phạm Quỳnh Hoa", "Thạc sĩ Bác sĩ", "Phòng 202 - Khu B", "Kinh nghiệm điều trị rối loạn giấc ngủ và đau dây thần kinh tọa."),
            # Khoa Tai Mũi Họng
            (dept_map["TAI_MUI_HONG"], "BS.CKII Vũ Đức Thịnh", "Bác sĩ Chuyên khoa II", "Phòng 301 - Khu C", "Chuyên nội soi ống mềm Tai Mũi Họng, nạo VA, cắt amidan, viêm tai giữa."),
            (dept_map["TAI_MUI_HONG"], "ThS.BS Đặng Thu Hương", "Thạc sĩ Bác sĩ", "Phòng 302 - Khu C", "Chuyên khám và điều trị viêm mũi dị ứng, viêm xoang mạn tính và polyp mũi."),
            # Khoa Tiêu hóa
            (dept_map["TIEU_HOA"], "BS.CKII Hoàng Minh Tuấn", "Bác sĩ Chuyên khoa II", "Phòng 401 - Khu D", "Nội soi tiêu hóa không đau, tầm soát ung thư đường tiêu hóa và trào ngược dạ dày."),
            (dept_map["TIEU_HOA"], "ThS.BS Ngô Thị Bích", "Thạc sĩ Bác sĩ", "Phòng 402 - Khu D", "Chuyên sâu hội chứng ruột kích thích (IBS), bệnh lý gan mật tụy và viêm loét dạ dày."),
            # Khoa Hô hấp
            (dept_map["HO_HAP"], "TS.BS Đỗ Quốc Cường", "Tiến sĩ Bác sĩ", "Phòng 501 - Khu E", "Chuyên gia thăm dò chức năng hô hấp, điều trị hen phế quản và bệnh phổi tắc nghẽn (COPD)."),
            (dept_map["HO_HAP"], "BS.CKI Bùi Ánh Nguyệt", "Bác sĩ Chuyên khoa I", "Phòng 502 - Khu E", "Chuyên trị ho dai dẳng, viêm phế quản co thắt và phục hồi chức năng phổi."),
        ]
        sql_doc = "INSERT INTO doctors (department_id, full_name, degree, room_number, bio) VALUES (%s, %s, %s, %s, %s);"
        cursor.executemany(sql_doc, doctors)
        print("✅ Đã nạp 12 Bác sĩ.")

        # Lấy ID của các bác sĩ vừa tạo
        cursor.execute("SELECT id FROM doctors;")
        doc_ids = [row[0] for row in cursor.fetchall()]

        # 3. Thêm Lịch khám (Schedules) cho 7 ngày tới
        today = date.today()
        time_slots = [
            "08:00 - 09:00",
            "09:30 - 10:30",
            "13:30 - 14:30",
            "15:00 - 16:00",
        ]

        schedules = []
        for doc_id in doc_ids:
            for day_offset in range(1, 8):
                work_date = today + timedelta(days=day_offset)
                for slot in time_slots:
                    max_p = 5
                    # Giả lập vài slot ngẫu nhiên đã có 1-2 bệnh nhân đặt
                    curr_p = random.choice([0, 1, 2, 5]) if random.random() < 0.25 else 0
                    is_avail = 1 if curr_p < max_p else 0
                    schedules.append((doc_id, work_date, slot, max_p, curr_p, is_avail))

        sql_sched = "INSERT INTO schedules (doctor_id, work_date, time_slot, max_patients, current_patients, is_available) VALUES (%s, %s, %s, %s, %s, %s);"
        cursor.executemany(sql_sched, schedules)
        print(f"✅ Đã tạo {len(schedules)} khung giờ khám cho 7 ngày tới.")
        print("🎉 Nạp dữ liệu vào MySQL thành công mỹ mãn!")

    except Exception as e:
        print(f"❌ Có lỗi xảy ra: {e}")
        conn.rollback()
    finally:
        cursor.close()
        conn.close()


if __name__ == "__main__":
    seed_mysql_data()

