# CTPN implement by C++

Ref: https://github.com/tianzhi0549/CTPN 

This project impliments the CTPN with C++

## build
cd ctpn-cpp
mkdir build
cd build
cmake ..
make

## the finished and developing work
We have finished the network till the TextDetector::detect() in src/detectors.py Line 51.
The remained work is to connect the text patch after nms to form the textline, see:
* TextDetector::detect() in src/detectors.py Line 52 -> text_lines=self.text_proposal_connector.get_text_lines(text_proposals, scores, im.shape[:2])
* src/text_proposal_connector.py -> whole file and its dependences
