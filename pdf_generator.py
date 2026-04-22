"""
PDF Generator for FAR/DAR Accident Reports.

Generates custom boxed FAR/DAR forms (non-template), preserving all major
form sections and yes/no columns while remaining easy to fill.
"""

import io
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak


def _val(data: dict, key: str, default: str = "") -> str:
    v = data.get(key)
    return str(v) if v not in (None, "") else default


def _yes_no_mark(value: str) -> str:
    v = str(value or "").strip().lower()
    yes = {"yes", "y", "true", "1", "available", "present", "found"}
    no = {"no", "n", "false", "0", "not available", "absent", "not found"}
    if v in yes:
        return "Yes [X]   No [ ]"
    if v in no:
        return "Yes [ ]   No [X]"
    return "Yes [ ]   No [ ]"


def _doc_flags(value: str) -> tuple[str, str]:
    v = str(value or "").strip().lower()
    yes = {"yes", "y", "true", "1", "attached"}
    no = {"no", "n", "false", "0", "not attached"}
    if v in yes:
        return "X", ""
    if v in no:
        return "", "X"
    return "", ""


def _styles():
    s = getSampleStyleSheet()
    s.add(ParagraphStyle("GovTitle", parent=s["Title"], alignment=TA_CENTER, fontSize=14))
    s.add(ParagraphStyle("GovSub", parent=s["Normal"], alignment=TA_CENTER, fontSize=9))
    s.add(ParagraphStyle("Section", parent=s["Heading4"], fontSize=10, spaceBefore=4 * mm, spaceAfter=2 * mm))
    s.add(ParagraphStyle("Cell", parent=s["Normal"], fontSize=8))
    return s


def _table(data, widths, header_rows=0):
    t = Table(data, colWidths=widths, repeatRows=header_rows)
    style = [
        ("GRID", (0, 0), (-1, -1), 0.6, colors.black),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    if header_rows:
        style += [
            ("FONTNAME", (0, 0), (-1, header_rows - 1), "Helvetica-Bold"),
            ("BACKGROUND", (0, 0), (-1, header_rows - 1), colors.HexColor("#f2f2f2")),
        ]
    t.setStyle(TableStyle(style))
    return t


def generate_far_pdf(data: dict) -> bytes:
    st = _styles()
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=12 * mm, rightMargin=12 * mm, topMargin=10 * mm, bottomMargin=10 * mm)
    el = []

    el.append(Paragraph("FORM 1 - FIRST ACCIDENT REPORT (FAR)", st["GovTitle"]))
    el.append(Paragraph("Custom digital format with full boxed sections", st["GovSub"]))
    el.append(Spacer(1, 3 * mm))

    el.append(Paragraph("A. FIR / Registration", st["Section"]))
    el.append(_table([
        ["FIR No.", _val(data, "fir_no"), "Date", _val(data, "report_date")],
        ["Under Section", _val(data, "ipc_sections"), "Police Station", _val(data, "police_station")],
    ], [30 * mm, 58 * mm, 30 * mm, 58 * mm]))

    el.append(Paragraph("B. Accident Details", st["Section"]))
    el.append(_table([
        ["1. Date of Accident", _val(data, "date_of_accident"), "2. Time", _val(data, "time_of_accident")],
        ["3. Place of Accident", _val(data, "place_of_accident"), "", ""],
        ["4. Source of Information", _val(data, "info_source"), "Informant", _val(data, "informant_name")],
        ["5. Nature of Accident", _val(data, "nature_of_accident"), "No. of Vehicles", _val(data, "num_vehicles", "0")],
        ["6. Fatalities", _val(data, "num_fatalities", "0"), "7. Injured", _val(data, "num_injured", "0")],
    ], [40 * mm, 48 * mm, 35 * mm, 53 * mm]))

    el.append(Paragraph("C. Yes/No Boxes", st["Section"]))
    el.append(_table([
        ["Registration number known", _yes_no_mark(_val(data, "reg_known"))],
        ["Vehicle impounded by police", _yes_no_mark(_val(data, "vehicles_impounded"))],
        ["Driver found on spot", _yes_no_mark(_val(data, "drivers_found"))],
        ["CCTV available", _yes_no_mark(_val(data, "cctv"))],
    ], [72 * mm, 104 * mm]))

    el.append(Paragraph("D. Vehicle Details (All boxes present)", st["Section"]))
    el.append(_table([
        ["Field", "Vehicle 1", "Vehicle 2"],
        ["Type", _val(data, "v1_type"), _val(data, "v2_type")],
        ["Registration No.", _val(data, "v1_reg"), _val(data, "v2_reg")],
        ["Driver Name", _val(data, "v1_driver"), _val(data, "v2_driver")],
        ["Owner Name", _val(data, "v1_owner"), _val(data, "v2_owner")],
        ["Insurance Company", _val(data, "v1_insurance"), _val(data, "v2_insurance")],
    ], [42 * mm, 67 * mm, 67 * mm], header_rows=1))

    el.append(PageBreak())
    el.append(Paragraph("E. Investigation / Medical", st["Section"]))
    el.append(_table([
        ["Collision Type", _val(data, "collision_type"), "Cause", _val(data, "cause")],
        ["Road Type", _val(data, "road_type"), "Weather", _val(data, "weather")],
        ["Lighting", _val(data, "lighting"), "Visibility", _val(data, "visibility")],
        ["Area/Jurisdiction", _val(data, "jurisdiction"), "Load Condition", _val(data, "load_condition")],
        ["Hospital", _val(data, "hospital"), "Doctor", _val(data, "doctor")],
    ], [35 * mm, 53 * mm, 35 * mm, 53 * mm]))

    el.append(Paragraph("F. Officer Verification", st["Section"]))
    el.append(_table([
        ["Investigating Officer", _val(data, "officer_name")],
        ["PIS/Employee No.", _val(data, "officer_pis")],
        ["Phone No.", _val(data, "officer_phone")],
        ["Police Station", _val(data, "police_station")],
        ["Date", _val(data, "report_date")],
        ["Signature", ""],
    ], [60 * mm, 116 * mm]))

    doc.build(el)
    return buf.getvalue()


def generate_dar_pdf(data: dict) -> bytes:
    st = _styles()
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=10 * mm, rightMargin=10 * mm, topMargin=10 * mm, bottomMargin=10 * mm)
    el = []

    el.append(Paragraph("FORM VII - DETAILED ACCIDENT REPORT (DAR)", st["GovTitle"]))
    el.append(Paragraph("Custom digital format with complete boxed layout", st["GovSub"]))
    el.append(Spacer(1, 2 * mm))

    # Page 1
    el.append(Paragraph("A. Case Registration", st["Section"]))
    el.append(_table([
        ["FIR No.", _val(data, "fir_no"), "Date", _val(data, "report_date")],
        ["Under Section", _val(data, "ipc_sections"), "Police Station", _val(data, "police_station")],
    ], [28 * mm, 60 * mm, 28 * mm, 60 * mm]))

    el.append(Paragraph("B. Accident Particulars (1-4)", st["Section"]))
    el.append(_table([
        ["1. Date of Accident", _val(data, "date_of_accident")],
        ["2. Time of Accident", _val(data, "time_of_accident")],
        ["3. Place of Accident", _val(data, "place_of_accident")],
        ["4. Nature of Accident", _val(data, "nature_of_accident")],
    ], [52 * mm, 128 * mm]))

    el.append(Paragraph("C. Offending Vehicle Details (5-9)", st["Section"]))
    el.append(_table([
        ["Field", "Vehicle 1", "Vehicle 2"],
        ["Registration No.", _val(data, "v1_reg"), _val(data, "v2_reg")],
        ["Make", _val(data, "v1_make"), _val(data, "v2_make")],
        ["Model", _val(data, "v1_model"), _val(data, "v2_model")],
        ["Vehicle Type", _val(data, "v1_type"), _val(data, "v2_type")],
        ["Vehicle Use Type", _val(data, "v1_use_type"), _val(data, "v2_use_type")],
        ["Driver Name", _val(data, "v1_driver"), _val(data, "v2_driver")],
        ["Owner Name", _val(data, "v1_owner"), _val(data, "v2_owner")],
        ["Insurance Company", _val(data, "v1_insurance"), _val(data, "v2_insurance")],
        ["9. Licence Verified", _yes_no_mark(_val(data, "license_verified_v1")), _yes_no_mark(_val(data, "license_verified_v2"))],
    ], [44 * mm, 66 * mm, 66 * mm], header_rows=1))

    el.append(PageBreak())

    # Page 2
    el.append(Paragraph("D. Driver Compliance Checks (10-19)", st["Section"]))
    el.append(_table([
        ["10. Licence suspended/cancelled", _yes_no_mark(_val(data, "license_suspended"))],
        ["11. Driver injured in accident", _yes_no_mark(_val(data, "driver_injured"))],
        ["12. Vehicle driven by", _val(data, "driven_by")],
        ["13. Under influence of alcohol/drugs", _yes_no_mark(_val(data, "alcohol"))],
        ["14. Carrying mobile while driving", _yes_no_mark(_val(data, "mobile"))],
        ["15. Previously involved in accident case", _yes_no_mark(_val(data, "previous_case"))],
        ["16. Commercial vehicle permit details", _val(data, "permit_details")],
        ["17. Permit and fitness verified", _yes_no_mark(_val(data, "permit_fitness_verified"))],
        ["18. Owner reported to insurer", _yes_no_mark(_val(data, "owner_reported_insurance"))],
        ["19. Driver fled from spot", _yes_no_mark(_val(data, "driver_fled"))],
    ], [72 * mm, 108 * mm]))

    el.append(Paragraph("E. Victim Details (20-29)", st["Section"]))
    el.append(_table([
        ["20. Victim Type", _val(data, "victim_type")],
        ["21. Name of deceased", _val(data, "deceased_name")],
        ["22. Age of deceased", _val(data, "deceased_age")],
        ["23. Occupation", _val(data, "deceased_occupation")],
        ["24. Legal representatives (summary)", _val(data, "legal_representatives")],
        ["25. Name of injured", _val(data, "injured_name")],
        ["26. Age", _val(data, "injured_age")],
        ["27. Occupation", _val(data, "injured_occupation")],
        ["28. Nature of Injury", _val(data, "injury_nature")],
        ["29. Injury Details", _val(data, "injury_details")],
    ], [72 * mm, 108 * mm]))

    el.append(PageBreak())

    # Page 3
    el.append(Paragraph("F. Offences Charged (30)", st["Section"]))
    ipc = _val(data, "ipc_sections")
    el.append(_table([
        ["Indian Penal Code / BNS sections", ipc],
        ["Motor Vehicles Act sections", _val(data, "mv_sections")],
    ], [72 * mm, 108 * mm]))

    ipc_rows = [
        ["a", "Section 279 - Rash driving or riding on a public way", "X" if "279" in ipc else ""],
        ["b", "Section 337 - Causing hurt by act endangering life", "X" if "337" in ipc else ""],
        ["c", "Section 338 - Causing grievous hurt", "X" if "338" in ipc else ""],
        ["d", "Section 304-A - Causing death by negligence", "X" if ("304-A" in ipc or "304A" in ipc) else ""],
        ["e", "Any other offence", _val(data, "other_offence")],
    ]
    el.append(_table([["Item", "IPC Offence", "Marked / Value"]] + ipc_rows, [12 * mm, 132 * mm, 36 * mm], header_rows=1))

    mv_text = _val(data, "mv_sections")
    mv_rows = [
        ["a", "3/181 Driving without license"],
        ["b", "4/181 Driving by minor"],
        ["c", "5/180 Allowing unauthorized person to drive"],
        ["d", "182 Offences relating to licenses"],
        ["e", "56/192 Without fitness"],
        ["f", "66(1)/192A Without permit"],
        ["g", "112/183(1) Over speeding"],
        ["h", "113/194 Over loading"],
        ["i", "119/184 Jumping red light"],
        ["j", "119/177 Violation of mandatory signs"],
        ["k", "122/177 Improper/obstructive parking"],
        ["l", "146/196 Without insurance"],
        ["m", "177/RRR 17(1) Violation of one way"],
        ["n", "194(1A)/RRR 29 Carrying high/long load"],
        ["o", "184/RRR rule 6 Violation of no overtaking"],
        ["p", "177/CMVR Rule 105 Without light after sunset"],
        ["q", "179 Disobedience/refusal of information"],
        ["r", "184 Driving dangerously"],
        ["s", "184 Using mobile phone while driving"],
        ["t", "185 Drunken driving/drugs"],
        ["u", "186 Driving unfit to drive"],
        ["v", "187 Violation of Sections 132(1)(a), 133, 134"],
        ["w", "190 Using vehicle in unsafe condition"],
        ["x", "194A Carrying more passengers than authorized"],
        ["y", "194B/CMVR Rule 138(3) Without safety belt"],
        ["z", "194C Motorcycle safety violation"],
        ["aa", "194D Not wearing protective headgear"],
        ["bb", "194E Not allowing emergency vehicles"],
        ["cc", "194F Horn misuse"],
        ["dd", "197 Taking vehicle without authority"],
        ["ee", "199A Offence by juveniles"],
        ["ff", "Any other offence"],
    ]
    el.append(Spacer(1, 2 * mm))
    el.append(Paragraph("Motor Vehicles Act, 1988 Offence Matrix", st["Section"]))
    mv_table = [["Item", "Offence", "Marked"]]
    for item, desc in mv_rows:
        key = desc.split(" ")[0]
        mv_table.append([item, desc, "X" if key in mv_text else ""])
    el.append(_table(mv_table, [14 * mm, 130 * mm, 36 * mm], header_rows=1))

    el.append(Paragraph("G. Narrative and Tribunal Directions (31-32)", st["Section"]))
    el.append(_table([
        ["31. Detailed description of accident", _val(data, "accident_description", _val(data, "cause"))],
        ["32. Directions required from Claims Tribunal", _val(data, "tribunal_directions")],
    ], [72 * mm, 108 * mm]))

    el.append(PageBreak())
    el.append(Paragraph("H. Standard Directions to Tribunal", st["Section"]))
    el.append(_table([
        ["i.", "Driver has not furnished / incomplete Form-III. Direction requested to submit within 15 days.", _val(data, "dir_i")],
        ["ii.", "Owner has not furnished / incomplete Form-IV. Direction requested to submit within 15 days.", _val(data, "dir_ii")],
        ["iii.", "Victim has not furnished / incomplete Form-VI / VIA. Direction requested to submit within 15 days.", _val(data, "dir_iii")],
        ["iv.", "Registration Authority verification report pending. Direction requested to submit within 15 days.", _val(data, "dir_iv")],
        ["v.", "Hospital MLC/Post Mortem report pending. Direction requested to submit within 15 days.", _val(data, "dir_v")],
    ], [10 * mm, 122 * mm, 48 * mm]))

    el.append(Paragraph("H. Documents Checklist (33)", st["Section"]))
    docs = [
        "i. FIR",
        "ii. Form-I - First Accident Report (FAR)",
        "iii. Form-II - Rights of Victim(s) and Flow Chart",
        "iv. Form-III - Driver's Form along with documents",
        "v. Form-IV - Owner's Form along with documents",
        "vi. Form-V - Interim Accident Report (IAR)",
        "vii. Form-VI - Victim's Form",
        "viii. Form-VIA - Minor children details",
        "ix. Form-VII - Detailed Accident Report (DAR)",
        "x. Form-VIII - Site Plan",
        "xi. Form-IX - Mechanical Inspection Report",
        "xii. Form-X - Verification Report",
        "xiii. Form-XI - Insurance Form",
        "xiv. Scene photographs (all angles)",
        "xv. Vehicle photographs (all angles)",
        "xvi. CCTV footage",
        "xvii. Report under section 173 CrPC",
        "xviii. Notice under section 133 MVA",
        "xix. Post-Mortem Report (Death Case)",
        "xx. MLC (Injury Case)",
        "xxi. Multi-angle photographs of injured",
        "xxii. Letter to driver",
        "xxiii. Letter to owner",
        "xxiv. Letter to insurance company",
        "xxv. Letter to victim(s)",
        "xxvi. Letter to registration authorities",
        "xxvii. Letter to hospital",
    ]
    chk = [["Document", "Attached", "Not Attached"]]
    for d in docs:
        a, na = _doc_flags(_val(data, f"doc_{d.lower().replace(' ', '_')}", ""))
        chk.append([d, a, na])
    el.append(_table(chk, [120 * mm, 30 * mm, 30 * mm], header_rows=1))

    el.append(Spacer(1, 4 * mm))
    el.append(PageBreak())
    el.append(Paragraph("I. Verification", st["Section"]))
    el.append(_table([
        ["Investigating Officer (S.H.O / I.O)", _val(data, "officer_name")],
        ["P.I.S / Employee No.", _val(data, "officer_pis")],
        ["Phone No.", _val(data, "officer_phone")],
        ["P.S.", _val(data, "police_station")],
        ["Date", _val(data, "report_date")],
        ["Signature", ""],
    ], [72 * mm, 108 * mm]))

    doc.build(el)
    return buf.getvalue()


if __name__ == "__main__":
    # Quick test
    test_data = {
        'fir_no': 'RJ/PS/045/2026',
        'date_of_accident': '18 April 2026',
        'time_of_accident': '3:30 PM',
        'place_of_accident': 'State Highway 61 near Mathania Bus Stand',
        'state': 'Rajasthan',
        'district': 'Jodhpur',
        'nature_of_accident': 'Fatal',
        'num_fatalities': '5',
        'num_injured': '36',
        'num_vehicles': '2',
        'v1_type': 'Passenger Bus',
        'v1_reg': 'RJ 19 PB 5678',
        'v2_type': 'Truck',
        'v2_reg': 'RJ 14 GA 1234',
        'collision_type': 'Head-On Collision',
        'cause': 'Overspeeding, Dangerous Overtaking',
        'weather': 'Clear',
        'road_type': 'State Highway',
        'officer_name': 'SI Mahendra Singh',
        'report_date': '22 April 2026',
    }

    far_pdf = generate_far_pdf(test_data)
    with open("test_far.pdf", "wb") as f:
        f.write(far_pdf)
    print(f"FAR PDF generated: {len(far_pdf):,} bytes")

    dar_pdf = generate_dar_pdf(test_data)
    with open("test_dar.pdf", "wb") as f:
        f.write(dar_pdf)
    print(f"DAR PDF generated: {len(dar_pdf):,} bytes")
