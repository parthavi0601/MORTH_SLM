import sys; sys.path.insert(0, '.')
from pytorch_qa import extract_from_text_regex

text = 'A total of five persons died and approximately thirty-six were injured.'
data = extract_from_text_regex(text)
print(f"Killed: {data.get('num_fatalities', '?')}")
print(f"Injured: {data.get('num_injured', '?')}")

# Full test
text2 = """FIR number RJ/PS/045/2026 was registered on 22 April 2026 at Mathania Police Station, Jodhpur under Sections 279 and 304A of IPC. The accident occurred on Friday, 18 April 2026 at 3:30 PM on State Highway 61 near Mathania Bus Stand in Jodhpur, Rajasthan. The source of information was police, and the informant was Constable Ramesh Kumar, residing at Mathania Police Station. The accident was fatal in nature and involved two vehicles. A total of five persons died and approximately thirty-six were injured."""

data2 = extract_from_text_regex(text2)
print(f"\nFull test:")
print(f"  FIR: {data2.get('fir_no', '?')}")
print(f"  Date: {data2.get('date_of_accident', '?')}")
print(f"  Killed: {data2.get('num_fatalities', '?')}")
print(f"  Injured: {data2.get('num_injured', '?')}")
print(f"  Cause: {data2.get('cause', '?')}")
