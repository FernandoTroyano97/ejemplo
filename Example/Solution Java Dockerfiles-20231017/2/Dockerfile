FROM amazoncorretto:19-alpine-jdk

COPY HelloWorld.java /app/

WORKDIR /app/

RUN javac HelloWorld.java

ENTRYPOINT ["java", "HelloWorld"]
