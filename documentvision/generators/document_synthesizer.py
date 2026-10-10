"""
Programmatic Document Synthesizer using Pillow.
Generates realistic high-resolution document images for invoices, tables, charts, and forms.
"""

import os
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, Any


class DocumentSynthesizer:
    """Renders crisp, high-resolution document benchmark images."""

    @staticmethod
    def _get_font(size: int = 14):
        try:
            # Try Windows standard fonts first
            font_path = "C:/Windows/Fonts/arial.ttf"
            if os.path.exists(font_path):
                return ImageFont.truetype(font_path, size)
            return ImageFont.load_default()
        except Exception:
            return ImageFont.load_default()

    @classmethod
    def render_invoice(cls, output_path: str) -> str:
        """Renders realistic corporate invoice document."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        img = Image.new("RGB", (1000, 1300), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)

        title_font = cls._get_font(28)
        header_font = cls._get_font(18)
        body_font = cls._get_font(14)
        bold_font = cls._get_font(15)

        # Header Banner
        draw.rectangle([50, 40, 950, 120], fill=(240, 244, 248), outline=(200, 215, 230), width=2)
        draw.text((70, 55), "ACME LOGISTICS & CLOUD SERVICES", fill=(30, 41, 59), font=title_font)
        draw.text((70, 92), "Enterprise Software Solutions • Tax ID: US-984729103", fill=(100, 116, 139), font=body_font)
        draw.text((780, 65), "INVOICE", fill=(71, 85, 105), font=title_font)

        # Metadata Box
        draw.rectangle([50, 140, 500, 260], fill=(248, 250, 252), outline=(226, 232, 240), width=1)
        draw.text((65, 150), "BILLED TO:", fill=(100, 116, 139), font=bold_font)
        draw.text((65, 175), "Nexus FinTech Corp.", fill=(15, 23, 42), font=bold_font)
        draw.text((65, 198), "Attn: Accounts Payable", fill=(51, 65, 85), font=body_font)
        draw.text((65, 220), "100 Montgomery St, Suite 1200", fill=(51, 65, 85), font=body_font)
        draw.text((65, 240), "San Francisco, CA 94104", fill=(51, 65, 85), font=body_font)

        draw.rectangle([520, 140, 950, 260], fill=(248, 250, 252), outline=(226, 232, 240), width=1)
        draw.text((535, 150), "INVOICE DETAILS:", fill=(100, 116, 139), font=bold_font)
        draw.text((535, 175), "Invoice Number: INV-2026-8841", fill=(15, 23, 42), font=body_font)
        draw.text((535, 198), "Issue Date: October 01, 2026", fill=(51, 65, 85), font=body_font)
        draw.text((535, 220), "Due Date: October 31, 2026", fill=(51, 65, 85), font=body_font)
        draw.text((535, 240), "PO Number: PO-77291", fill=(51, 65, 85), font=body_font)

        # Table Header
        draw.rectangle([50, 290, 950, 330], fill=(30, 41, 59))
        draw.text((70, 300), "Item / Description", fill=(255, 255, 255), font=bold_font)
        draw.text((520, 300), "Qty", fill=(255, 255, 255), font=bold_font)
        draw.text((630, 300), "Unit Price", fill=(255, 255, 255), font=bold_font)
        draw.text((820, 300), "Total", fill=(255, 255, 255), font=bold_font)

        # Table Rows
        items = [
            ("Dedicated GPU Cluster Instance (A100-80G x 8)", "4 mos", "$8,500.00", "$34,000.00"),
            ("High-Throughput Vector DB Tier (Enterprise)", "1 yr", "$12,400.00", "$12,400.00"),
            ("Automated Data Quality Annotation Pipeline", "1 lic", "$4,850.00", "$4,850.00"),
            ("Dedicated Technical Support & SLA 99.99%", "12 mos", "$750.00", "$9,000.00"),
            ("Security & Compliance Penetration Audit", "1 svcs", "$6,250.00", "$6,250.00"),
        ]

        y_pos = 345
        for i, (desc, qty, unit, total) in enumerate(items):
            bg_col = (255, 255, 255) if i % 2 == 0 else (248, 250, 252)
            draw.rectangle([50, y_pos - 8, 950, y_pos + 26], fill=bg_col, outline=(226, 232, 240), width=1)
            draw.text((70, y_pos), desc, fill=(30, 41, 59), font=body_font)
            draw.text((520, y_pos), qty, fill=(51, 65, 85), font=body_font)
            draw.text((630, y_pos), unit, fill=(51, 65, 85), font=body_font)
            draw.text((820, y_pos), total, fill=(15, 23, 42), font=bold_font)
            y_pos += 42

        # Summary Totals Box
        summary_y = y_pos + 40
        draw.rectangle([550, summary_y, 950, summary_y + 160], fill=(241, 245, 249), outline=(203, 213, 225), width=2)
        draw.text((570, summary_y + 15), "Subtotal:", fill=(71, 85, 105), font=body_font)
        draw.text((830, summary_y + 15), "$66,500.00", fill=(30, 41, 59), font=body_font)

        draw.text((570, summary_y + 45), "Tax (8.25%):", fill=(71, 85, 105), font=body_font)
        draw.text((830, summary_y + 45), "$5,486.25", fill=(30, 41, 59), font=body_font)

        draw.text((570, summary_y + 75), "Credits Applied:", fill=(71, 85, 105), font=body_font)
        draw.text((830, summary_y + 75), "-$1,000.00", fill=(16, 185, 129), font=body_font)

        draw.line([570, summary_y + 105, 930, summary_y + 105], fill=(203, 213, 225), width=2)
        draw.text((570, summary_y + 120), "TOTAL DUE:", fill=(15, 23, 42), font=title_font)
        draw.text((790, summary_y + 120), "$70,986.25", fill=(30, 58, 138), font=title_font)

        # Payment Instructions & Signature
        draw.text((70, summary_y + 30), "PAYMENT TERMS: Net 30 Days", fill=(100, 116, 139), font=bold_font)
        draw.text((70, summary_y + 55), "Wire Transfer: Bank of America, NY", fill=(51, 65, 85), font=body_font)
        draw.text((70, summary_y + 75), "Account Number: 0098-4491-8821", fill=(51, 65, 85), font=body_font)
        draw.text((70, summary_y + 95), "Routing Number: 026009593", fill=(51, 65, 85), font=body_font)

        draw.rectangle([70, summary_y + 140, 350, summary_y + 200], fill=(255, 255, 255), outline=(203, 213, 225), width=1)
        draw.text((80, summary_y + 145), "Authorized Signature:", fill=(148, 163, 184), font=cls._get_font(11))
        draw.text((120, summary_y + 170), "Elena Rostova, VP Finance", fill=(30, 41, 59), font=bold_font)

        img.save(output_path, "PNG")
        return output_path

    @classmethod
    def render_balance_sheet(cls, output_path: str) -> str:
        """Renders financial balance sheet table image with nested structures."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        img = Image.new("RGB", (1000, 950), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)

        title_font = cls._get_font(24)
        bold_font = cls._get_font(15)
        body_font = cls._get_font(14)

        # Header
        draw.text((50, 40), "GLOBAL TECHNOLOGY HOLDINGS INC.", fill=(15, 23, 42), font=title_font)
        draw.text((50, 75), "CONSOLIDATED STATEMENTS OF FINANCIAL POSITION (BALANCE SHEET)", fill=(71, 85, 105), font=bold_font)
        draw.text((50, 98), "As of September 30, 2026 and December 31, 2025 • (Amounts in USD Millions)", fill=(100, 116, 139), font=body_font)

        # Table Header
        draw.rectangle([50, 135, 950, 175], fill=(30, 41, 59))
        draw.text((70, 145), "Assets & Liabilities Category", fill=(255, 255, 255), font=bold_font)
        draw.text((620, 145), "Q3 2026 (Audited)", fill=(255, 255, 255), font=bold_font)
        draw.text((810, 145), "FY 2025 (Audited)", fill=(255, 255, 255), font=bold_font)

        rows = [
            ("Current Assets:", "", "", True),
            ("  Cash and cash equivalents", "$4,821.5", "$3,940.2", False),
            ("  Short-term marketable securities", "$2,150.0", "$2,400.0", False),
            ("  Accounts receivable, net", "$1,680.4", "$1,410.8", False),
            ("  Inventories and supplies", "$845.2", "$790.0", False),
            ("Total Current Assets", "$9,497.1", "$8,541.0", True),
            ("Non-Current Assets:", "", "", True),
            ("  Property, plant and equipment, net", "$8,210.0", "$7,650.0", False),
            ("  Goodwill and intangible assets", "$3,450.0", "$3,500.0", False),
            ("Total Non-Current Assets", "$11,660.0", "$11,150.0", True),
            ("TOTAL ASSETS", "$21,157.1", "$19,691.0", True),
            ("Current Liabilities:", "", "", True),
            ("  Accounts payable and accrued expenses", "$2,340.5", "$2,110.0", False),
            ("  Short-term debt obligations", "$1,200.0", "$1,500.0", False),
            ("Total Current Liabilities", "$3,540.5", "$3,610.0", True),
            ("TOTAL LIABILITIES & STOCKHOLDERS' EQUITY", "$21,157.1", "$19,691.0", True),
        ]

        y_pos = 185
        for desc, q3, fy, is_bold in rows:
            bg_col = (241, 245, 249) if is_bold else (255, 255, 255)
            draw.rectangle([50, y_pos - 4, 950, y_pos + 26], fill=bg_col, outline=(226, 232, 240), width=1)
            font_to_use = bold_font if is_bold else body_font
            color_to_use = (15, 23, 42) if is_bold else (51, 65, 85)
            draw.text((70, y_pos), desc, fill=color_to_use, font=font_to_use)
            draw.text((640, y_pos), q3, fill=color_to_use, font=font_to_use)
            draw.text((830, y_pos), fy, fill=color_to_use, font=font_to_use)
            y_pos += 34

        img.save(output_path, "PNG")
        return output_path

    @classmethod
    def render_revenue_chart(cls, output_path: str) -> str:
        """Renders quarterly revenue chart with bars and trends."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        img = Image.new("RGB", (950, 750), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)

        title_font = cls._get_font(22)
        bold_font = cls._get_font(15)
        body_font = cls._get_font(13)

        # Title
        draw.text((50, 30), "Enterprise AI Cloud: Quarterly Revenue & Operating Margin (2025-2026)", fill=(15, 23, 42), font=title_font)
        draw.text((50, 65), "Data reported in Millions USD ($M) • Source: Q3 2026 Investor Report", fill=(100, 116, 139), font=body_font)

        # Plot Area: [120, 120] to [880, 580]
        x_min, x_max = 120, 880
        y_min, y_max = 120, 580

        # Axis Lines
        draw.line([x_min, y_min, x_min, y_max], fill=(148, 163, 184), width=2)
        draw.line([x_min, y_max, x_max, y_max], fill=(148, 163, 184), width=2)

        # Y-Grid lines (0 to 100M)
        for val in range(0, 101, 20):
            y_coord = y_max - int((val / 100.0) * (y_max - y_min))
            draw.line([x_min, y_coord, x_max, y_coord], fill=(241, 245, 249), width=1)
            draw.text((65, y_coord - 8), f"${val}M", fill=(100, 116, 139), font=body_font)

        # Data points (Quarter, Revenue, Profit)
        quarters = ["Q1-25", "Q2-25", "Q3-25", "Q4-25", "Q1-26", "Q2-26", "Q3-26"]
        revenues = [42.5, 48.0, 55.2, 68.4, 74.1, 82.5, 94.8]
        margins = [8.5, 11.2, 14.0, 20.1, 22.5, 27.0, 33.2]

        num_q = len(quarters)
        bar_width = 38
        gap = (x_max - x_min) / (num_q + 1)

        prev_margin_pt = None
        for i, (q, rev, marg) in enumerate(zip(quarters, revenues, margins)):
            center_x = x_min + int((i + 1) * gap)
            bar_height = int((rev / 100.0) * (y_max - y_min))
            bar_top = y_max - bar_height

            # Draw Revenue Bar
            draw.rectangle(
                [center_x - bar_width // 2, bar_top, center_x + bar_width // 2, y_max],
                fill=(67, 97, 238),
                outline=(30, 58, 138),
                width=1,
            )
            draw.text((center_x - 18, bar_top - 20), f"${rev:.1f}", fill=(30, 58, 138), font=bold_font)
            draw.text((center_x - 18, y_max + 12), q, fill=(71, 85, 105), font=bold_font)

            # Draw Profit Margin Point
            margin_height = int((marg / 100.0) * (y_max - y_min))
            margin_y = y_max - margin_height
            curr_margin_pt = (center_x, margin_y)

            if prev_margin_pt:
                draw.line([prev_margin_pt, curr_margin_pt], fill=(239, 68, 68), width=3)
            draw.ellipse([center_x - 5, margin_y - 5, center_x + 5, margin_y + 5], fill=(239, 68, 68))
            draw.text((center_x - 14, margin_y - 20), f"${marg:.1f}", fill=(239, 68, 68), font=body_font)

            prev_margin_pt = curr_margin_pt

        # Legend
        draw.rectangle([x_min, 640, x_max, 700], fill=(248, 250, 252), outline=(226, 232, 240), width=1)
        draw.rectangle([180, 660, 210, 680], fill=(67, 97, 238))
        draw.text((220, 662), "Gross Cloud Revenue ($M)", fill=(30, 41, 59), font=body_font)

        draw.line([520, 670, 560, 670], fill=(239, 68, 68), width=3)
        draw.ellipse([536, 666, 544, 674], fill=(239, 68, 68))
        draw.text((570, 662), "Operating Profit ($M)", fill=(30, 41, 59), font=body_font)

        img.save(output_path, "PNG")
        return output_path

    @classmethod
    def render_clinical_lab(cls, output_path: str) -> str:
        """Renders clinical laboratory diagnostic report document."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        img = Image.new("RGB", (1000, 1100), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)

        title_font = cls._get_font(24)
        bold_font = cls._get_font(15)
        body_font = cls._get_font(13)

        # Header
        draw.rectangle([50, 30, 950, 110], fill=(241, 245, 249), outline=(203, 213, 225), width=2)
        draw.text((70, 45), "METROPOLITAN UNIVERSITY HOSPITAL • PATHOLOGY LAB", fill=(15, 23, 42), font=title_font)
        draw.text((70, 80), "CLINICAL BIOCHEMISTRY & METABOLIC PANEL REPORT", fill=(71, 85, 105), font=bold_font)

        # Patient Info
        draw.rectangle([50, 130, 950, 210], fill=(248, 250, 252), outline=(226, 232, 240), width=1)
        draw.text((70, 145), "Patient Name: Marcus Vance", fill=(15, 23, 42), font=bold_font)
        draw.text((370, 145), "DOB: 1984-06-12 (Age: 42)", fill=(51, 65, 85), font=body_font)
        draw.text((670, 145), "MRN: #99482104", fill=(51, 65, 85), font=body_font)

        draw.text((70, 175), "Ordering Physician: Dr. Sarah Lin, MD", fill=(51, 65, 85), font=body_font)
        draw.text((370, 175), "Collection: 2026-10-08 07:30", fill=(51, 65, 85), font=body_font)
        draw.text((670, 175), "Reported: 2026-10-08 14:15", fill=(51, 65, 85), font=body_font)

        # Table Header
        draw.rectangle([50, 235, 950, 275], fill=(30, 41, 59))
        draw.text((70, 245), "Test Name", fill=(255, 255, 255), font=bold_font)
        draw.text((370, 245), "Result Value", fill=(255, 255, 255), font=bold_font)
        draw.text((540, 245), "Units", fill=(255, 255, 255), font=bold_font)
        draw.text((680, 245), "Reference Range", fill=(255, 255, 255), font=bold_font)
        draw.text((850, 245), "Flag", fill=(255, 255, 255), font=bold_font)

        tests = [
            ("Fasting Blood Glucose", "142", "mg/dL", "70 - 99", "HIGH ⚠️"),
            ("Hemoglobin A1c", "7.4", "%", "4.0 - 5.6", "HIGH ⚠️"),
            ("Serum Creatinine", "0.95", "mg/dL", "0.74 - 1.35", "NORMAL"),
            ("Blood Urea Nitrogen (BUN)", "18", "mg/dL", "6 - 20", "NORMAL"),
            ("Estimated GFR", "94", "mL/min/1.73m²", "> 60", "NORMAL"),
            ("Total Cholesterol", "228", "mg/dL", "< 200", "HIGH ⚠️"),
            ("HDL Cholesterol", "42", "mg/dL", "> 40", "NORMAL"),
            ("LDL Cholesterol", "148", "mg/dL", "< 100", "HIGH ⚠️"),
            ("Serum Potassium", "4.3", "mmol/L", "3.5 - 5.1", "NORMAL"),
            ("Serum Sodium", "139", "mmol/L", "135 - 145", "NORMAL"),
        ]

        y_pos = 285
        for name, res, unit, ref, flag in tests:
            is_high = "HIGH" in flag
            bg_col = (254, 242, 242) if is_high else (255, 255, 255)
            draw.rectangle([50, y_pos - 4, 950, y_pos + 26], fill=bg_col, outline=(226, 232, 240), width=1)
            draw.text((70, y_pos), name, fill=(30, 41, 59), font=body_font)
            val_col = (220, 38, 38) if is_high else (15, 23, 42)
            draw.text((370, y_pos), res, fill=val_col, font=bold_font)
            draw.text((540, y_pos), unit, fill=(71, 85, 105), font=body_font)
            draw.text((680, y_pos), ref, fill=(71, 85, 105), font=body_font)
            flag_col = (220, 38, 38) if is_high else (16, 185, 129)
            draw.text((850, y_pos), flag, fill=flag_col, font=bold_font)
            y_pos += 36

        img.save(output_path, "PNG")
        return output_path

    @classmethod
    def synthesize_all(cls, base_dir: str = "data/images") -> Dict[str, str]:
        os.makedirs(base_dir, exist_ok=True)
        paths = {
            "invoice": cls.render_invoice(os.path.join(base_dir, "invoice_001.png")),
            "balance_sheet": cls.render_balance_sheet(os.path.join(base_dir, "balance_sheet_001.png")),
            "revenue_chart": cls.render_revenue_chart(os.path.join(base_dir, "revenue_chart_001.png")),
            "clinical_lab": cls.render_clinical_lab(os.path.join(base_dir, "clinical_lab_001.png")),
        }
        return paths
