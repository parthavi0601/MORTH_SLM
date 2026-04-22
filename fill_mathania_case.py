"""
Runner script: fills FAR and DAR forms for the Mathania Jodhpur
head-on collision case (FIR RJ/PS/045/2026) using pdf_generator.
"""

import os
from pdf_generator import generate_far_pdf, generate_dar_pdf


data = {
    # General
    "fir_no": "RJ/PS/045/2026",
    "report_date": "22 April 2026",
    "ipc_sections": "279, 304-A IPC; 112/183(1), 184 MV Act",
    "police_station": "Mathania Police Station, Jodhpur, Rajasthan",

    # Accident details
    "date_of_accident": "18 April 2026 (Friday)",
    "time_of_accident": "3:30 PM",
    "place_of_accident": "State Highway 61, near Mathania Bus Stand, Jodhpur, Rajasthan",
    "nature_of_accident": "Fatal",
    "num_vehicles": "2",
    "num_fatalities": "5",
    "num_injured": "36",

    # Impoundment / drivers
    "vehicles_impounded": "Yes",
    "drivers_found": "Yes",

    # Hospital / medical
    "hospital": "Mathania Government Hospital",
    "doctor": "Dr. S. Sharma",
    "cctv": "No",

    # Vehicle 1 - Bus
    "v1_type": "Bus (Passenger / Public Service Vehicle)",
    "v1_reg": "RJ 19 PB 5678",
    "v1_driver": "Mohan Lal, resident of Jodhpur, Rajasthan",
    "v1_owner": "Rajasthan State Transport Corporation, Jodhpur",
    "v1_insurance": "National Insurance Company (policy under verification)",

    # Vehicle 2 - Truck
    "v2_type": "Truck / Lorry (Goods Carriage)",
    "v2_reg": "RJ 14 GA 1234",
    "v2_driver": "Raju Singh, resident of Jaipur, Rajasthan",
    "v2_owner": "Suresh Transport Company, Jaipur",
    "v2_insurance": "ICICI Lombard General Insurance (policy under verification)",

    # Driver / licensing facts (common to both)
    "license_verified": "No",                 # reports awaited
    "license_suspended": "No",
    "driver_injured": "No",
    "driven_by": "Paid driver",
    "alcohol": "No",
    "mobile": "No",
    "previous_case": "No",
    "permit_fitness_verified": "No",          # under verification
    "owner_reported_insurance": "Yes",        # owners directed to report
    "driver_fled": "No",

    # Victims / injuries / offence
    "injury_nature": "Grievous",
    "collision_type": "Head-on collision",
    "cause": "Overspeeding and dangerous overtaking",
    "weather": "Clear",
    "road_type": "State Highway (open stretch, Panchayat area)",

    # Officer
    "officer_name": "SI Mahendra Singh",
    "officer_pis": "4587",
    "officer_phone": "98XXXXXX21",
}


def main():
    out_dir = os.path.dirname(__file__)
    far_path = os.path.join(out_dir, "FAR_RJ_PS_045_2026.pdf")
    dar_path = os.path.join(out_dir, "DAR_RJ_PS_045_2026.pdf")

    far_bytes = generate_far_pdf(data)
    with open(far_path, "wb") as f:
        f.write(far_bytes)
    print(f"FAR written: {far_path} ({len(far_bytes):,} bytes)")

    dar_bytes = generate_dar_pdf(data)
    with open(dar_path, "wb") as f:
        f.write(dar_bytes)
    print(f"DAR written: {dar_path} ({len(dar_bytes):,} bytes)")


if __name__ == "__main__":
    main()
