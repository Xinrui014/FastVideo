
with open("/data/atlas/datasets/Inter4K/output_videos/descriptions.txt", "r", encoding="utf-8") as infile, open("output.txt", "w", encoding="utf-8") as outfile:
    for line in infile:
        # Remove the video ID and colon, then strip any extra whitespace
        prompt = line.partition(':')[2].strip()
        # Write the cleaned prompt to the output file with a newline
        outfile.write(prompt + "\n")