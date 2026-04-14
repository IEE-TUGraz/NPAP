import urllib.request

url = "https://zenodo.org/records/10952917/files/IRENASTAT_capacities_2000-2023.csv"
with urllib.request.urlopen(url) as resp:
    for _ in range(12):
        line = resp.readline().decode("utf-8", "replace")
        print(repr(line))
