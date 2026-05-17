plugins {
    application
    java
}

group = "cse8003.vsa"
version = "1.0.0"

java {
    toolchain {
        languageVersion.set(JavaLanguageVersion.of(17))
    }
}

application {
    mainClass.set("cse8003.vsa.appl.Main")
}

repositories {
    mavenCentral()
}

tasks.named<JavaExec>("run") {
    standardInput = System.`in`
}

tasks.test {
    useJUnitPlatform()
}

dependencies {
    implementation("org.apache.commons:commons-rng-client-api:1.6")
    testImplementation("org.junit.jupiter:junit-jupiter:5.10.2")
}
