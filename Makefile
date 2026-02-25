# Makefile

# List all cpp files in current directory
SRC = $(wildcard *.cpp)
# Binary folder
BIN = bin/myprogram

all: $(BIN)

# Compile all .cpp files into one binary
$(BIN): $(SRC)
	mkdir -p bin
	g++ $(SRC) -o $(BIN)

run: $(BIN)
	./$(BIN)

clean:
	rm -f bin/*