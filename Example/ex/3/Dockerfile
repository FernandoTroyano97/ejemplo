FROM amazoncorretto:19-alpine-jdk as build

WORKDIR /app/

COPY volume/HelloWorld.java /app/

RUN javac HelloWorld.java

FROM amazoncorretto:19-alpine-jdk as execute

WORKDIR /app/

COPY --from=build /app/HelloWorld.class /app/HelloWorld.class

ENTRYPOINT [ "java", "HelloWorld" ]


