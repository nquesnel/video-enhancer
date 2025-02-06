from src.enhancer import VideoEnhancer
import argparse

def main():
    parser = argparse.ArgumentParser(description='Enhance video quality')
    parser.add_argument('input', help='Input video path')
    parser.add_argument('output', help='Output video path')
    args = parser.parse_args()

    enhancer = VideoEnhancer()
    enhancer.enhance_video(args.input, args.output)

if __name__ == '__main__':
    main()