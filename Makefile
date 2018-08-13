CC=g++
EXEC=mimetik
BIN=/usr/local/bin

all: 
	$(CC) -o "$(EXEC)" main.cpp mimetik.cpp mimetik.h multilayerPerceptron.cpp multilayerPerceptron.h

clean:
	rm -rf $(EXEC)

install:
	cp -f "$(EXEC)" $(BIN)

