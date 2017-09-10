package ro.mnist.aspect;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.aspectj.lang.JoinPoint;
import org.aspectj.lang.Signature;
import org.aspectj.lang.annotation.AfterThrowing;
import org.aspectj.lang.annotation.Aspect;
import org.springframework.stereotype.Component;

import java.util.Arrays;

@Aspect
@Component
public class ExceptionHandlerAspect {

    private static Log logger = LogFactory.getLog(ExceptionHandlerAspect.class);

    @AfterThrowing(pointcut = "@annotation(ro.mnist.aspect.ExceptionHandler)", throwing = "e")
    public void exceptionHandler(JoinPoint joinPoint, Exception e) {

        Signature signature = joinPoint.getSignature();
        String methodName = signature.getName();
        String stuff = signature.toString();
        String arguments = Arrays.toString(joinPoint.getArgs());

        logger.error("We have caught exception in method: " + methodName + " with arguments "
                + arguments + "\nand the full toString: " + stuff + "\nthe exception is: "
                + e.getMessage(), e);

    }
}
