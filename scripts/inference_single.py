from engine.inference import predict_from_video


if __name__ == "__main__":
    mp4_path = input("Enter mp4 path: ").strip()
    mp4_path = mp4_path.replace("\ufeff", "").replace("ㅅ", "")

    if not mp4_path:
        raise SystemExit("No path provided.")

    score, level = predict_from_video(mp4_path)
    print(f"Score: {score:.2f}, Level: {level}")

