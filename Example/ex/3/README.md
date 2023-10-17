## Build

Build the image:

```console
C:\Users\antonio> docker build -t docker-compile-javac .
```

## Run

Run the following command to start the container:

```console
C:\Users\antonio> docker run -v %cd%/volume:/app docker-compile-javac
```