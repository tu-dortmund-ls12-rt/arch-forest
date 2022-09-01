from math import e
from ForestConverter import TreeConverter
import numpy as np
import heapq

import re


class HybridTreeConverter(TreeConverter):
    def __init__(self, dim, namespace, featureType):
        super().__init__(dim, namespace, featureType)

    def getArrayLenType(self, arrLen):
        arrayLenBit = int(np.log2(arrLen)) + 1
        if arrayLenBit <= 8:
            arrayLenDataType = "unsigned char"
        elif arrayLenBit <= 16:
            arrayLenDataType = "unsigned short"
        else:
            arrayLenDataType = "unsigned int"
        return arrayLenDataType

    def getImplementation(self, head, treeID):
        raise NotImplementedError(
            "This function should not be called directly, but only by a sub-class")

    def getHeader(self, splitType, treeID, arrLen, numClasses):
        dimBit = int(np.log2(self.dim)) + 1 if self.dim != 0 else 1

        if dimBit <= 8:
            dimDataType = "unsigned char"
        elif dimBit <= 16:
            dimDataType = "unsigned short"
        else:
            dimDataType = "unsigned int"

        featureType = self.getFeatureType()
        if (numClasses == 2):
            headerCode = """struct {namespace}_Node{treeID} {
                        //bool isLeaf;
                        //unsigned int prediction;
                        {dimDataType} feature;
                        {splitType} split;
                        {arrayLenDataType} leftChild;
                        {arrayLenDataType} rightChild;
                        unsigned char indicator;

                };\n""".replace("{namespace}", self.namespace) \
                .replace("{treeID}", str(treeID)) \
                .replace("{splitType}", splitType) \
                .replace("{dimDataType}", dimDataType) \
                .replace("{arrayLenDataType}", self.getArrayLenType(arrLen))
        else:
            headerCode = """struct {namespace}_Node{treeID} {
                           //bool isLeaf;
                            {dimDataType} feature;
                            {splitType} split;
                            {arrayLenDataType} leftChild;
                            {arrayLenDataType} rightChild;
                            unsigned char indicator;
                };\n""".replace("{namespace}", self.namespace) \
                .replace("{treeID}", str(treeID)) \
                .replace("{splitType}", splitType) \
                .replace("{dimDataType}", dimDataType) \
                .replace("{arrayLenDataType}", self.getArrayLenType(arrLen))

        headerCode += "inline unsigned int {namespace}_predict{treeID}({feature_t} const pX[{dim}]);\n" \
            .replace("{treeID}", str(treeID)) \
            .replace("{dim}", str(self.dim)) \
            .replace("{namespace}", self.namespace) \
            .replace("{feature_t}", featureType)
        return headerCode

    def getCode(self, tree, treeID, numClasses):
        # kh.chen
        # Note: this function has to be called once to traverse the tree to calculate the probabilities.
        tree.getProbAllPaths()
        cppCode, arrLen = self.getImplementation(tree.head, treeID)

        if self.containsFloat(tree):
            splitDataType = "float"
        else:
            lower, upper = self.getSplitRange(tree)

            bitUsed = 0
            if lower > 0:
                prefix = "unsigned"
                maxVal = upper
            else:
                prefix = ""
                bitUsed = 1
                maxVal = max(-lower, upper)

            splitBit = int(np.log2(maxVal) + 1 if maxVal != 0 else 1)

            if splitBit <= (8-bitUsed):
                splitDataType = prefix + " char"
            elif splitBit <= (16-bitUsed):
                splitDataType = prefix + " short"
            else:
                splitDataType = prefix + " int"
        headerCode = self.getHeader(splitDataType, treeID, arrLen, numClasses)

        return headerCode, cppCode

import ctypes
def float_to_bin(number):
    return "{0:#0{1}x}".format( (ctypes.c_uint.from_buffer(ctypes.c_float(number)).value) ,10)
def double_to_bin(number):
    return "{0:#0{1}x}".format( (ctypes.c_ulong.from_buffer(ctypes.c_double(number)).value) , 18)

def long_to_bin(number):
    return "{0:#0{1}x}".format( (ctypes.c_ulong.from_buffer(ctypes.c_long(number)).value) , 18)

def byte_width(dtype):
    if dtype==" char" or dtype=="unsigned char":
        return 1
    if dtype==" short" or dtype=="unsigned short":
        return 2
    if dtype==" int" or dtype=="unsigned int":
        return 4
    if dtype==" long" or dtype=="unsigned long":
        return 8
    if dtype=="float":
        return 4
    if dtype=="double":
        return 8

class IntLogicFloatHybridTreeConverter(HybridTreeConverter):
    def __init__(self, dim, namespace, featureType, architecture, segment_depth, start_native):
        super().__init__(dim, namespace, featureType)
        self.segment_depth = segment_depth
        self.start_native = start_native
        self.architecture = architecture

    def getHeader(self, splitType, treeID, arrLen, numClasses):
        dimBit = int(np.log2(self.dim)) + 1 if self.dim != 0 else 1

        if dimBit <= 8:
            dimDataType = "unsigned char"
        elif dimBit <= 16:
            dimDataType = "unsigned short"
        else:
            dimDataType = "unsigned int"

        headerCode=""

        featureType = self.getFeatureType()
        headerCode+="//Feature Type: "+str(featureType)+"\n"

        headerCode += "inline unsigned int {namespace}_predict{treeID}({feature_t} const pX[{dim}]);\n" \
            .replace("{treeID}", str(treeID)) \
            .replace("{dim}", str(self.dim)) \
            .replace("{namespace}", self.namespace) \
            .replace("{feature_t}", featureType)

        return headerCode

    def getImplementation(self, head, treeID):

        label_idx=[0]

        #Generate pure if else tree
        def write_tree(head, indent="", label_idx=label_idx):
            if head.prediction is not None:
                #Leaf
                #Move return value to return register and jump to end
                code=""
                if self.architecture == "intel":
                    code= indent+"\"mov $"+str(int(np.argmax(head.prediction)))+", %0;\"\n"
                    code+= indent+"\"jmp __rtitt_end_"+str(treeID)+";\"\n"
                else:
                    code= indent+"\"mov %0, #"+str(int(np.argmax(head.prediction)))+";\"\n"
                    code+= indent+"\"b __rtitt_end_"+str(treeID)+";\"\n"
                return code
            code=""

            #Normal node
            #Load feature value
            if self.getFeatureType() == "float":
                if self.architecture == "intel":
                    #code += indent+"\"movss "+str(head.feature*4)+"(%1), %%xmm1;\"\n"
                    #Load the integer representation instead
                    code += indent+"\"mov "+str(head.feature*4)+"(%1), %%ebx;\"\n"
                else:
                    code += indent+"\"ldrsw x1, [%1, "+str(head.feature*4)+"];\"\n"
            elif self.getFeatureType() == "double":
                if self.architecture == "intel":
                    # code += indent+"\"movsd "+str(head.feature*8)+"(%1), %%xmm1;\"\n"
                    #Load the integer representation instead
                    code += indent+"\"mov "+str(head.feature*8)+"(%1), %%rbx;\"\n"
                else:
                    code += indent+"\"ldr x1, [%1, "+str(head.feature*8)+"];\"\n"
            elif self.getFeatureType() == " char":
                if self.architecture == "intel":
                    code = indent+"\"mov "+str(head.feature*1)+"(%1), %%cl;\"\n"
                    code += indent+"\"movsx %%cl, %%rbx;\"\n"
                else:
                    code = indent+"\"ldrsb x1, [%1, "+str(head.feature*1)+"];\"\n"
            elif self.getFeatureType() == "unsigned char":
                if self.architecture == "intel":
                    code = indent+"\"mov $0, %%rbx;\"\n"
                    code += indent+"\"mov "+str(head.feature*1)+"(%1), %%bl;\"\n"
                else:
                    code = indent+"\"ldrb w1, [%1, "+str(head.feature*1)+"];\"\n"
            elif self.getFeatureType() == " short":
                if self.architecture == "intel":
                    code = indent+"\"mov "+str(head.feature*2)+"(%1), %%cx;\"\n"
                    code += indent+"\"movsx %%cx, %%rbx;\"\n"
                else:
                    code = indent+"\"ldrsh x1, [%1, "+str(head.feature*2)+"];\"\n"
            elif self.getFeatureType() == "unsigned short":
                if self.architecture == "intel":
                    code = indent+"\"mov $0, %%rbx;\"\n"
                    code += indent+"\"mov "+str(head.feature*2)+"(%1), %%bx;\"\n"
                else:
                    code = indent+"\"ldrh x1, [%1, "+str(head.feature*2)+"];\"\n"
            elif self.getFeatureType() == " int":
                if self.architecture == "intel":
                    code = indent+"\"mov "+str(head.feature*4)+"(%1), %%ecx;\"\n"
                    code += indent+"\"movsx %%ecx, %%rbx;\"\n"
                else:
                    code = indent+"\"ldrsw x1, [%1, "+str(head.feature*4)+"];\"\n"
            elif self.getFeatureType() == "unsigned int":
                if self.architecture == "intel":
                    code = indent+"\"mov $0, %%rbx;\"\n"
                    code += indent+"\"mov "+str(head.feature*4)+"(%1), %%ebx;\"\n"
                else:
                    code = indent+"\"ldrw x1, [%1, "+str(head.feature*4)+"];\"\n"
            elif self.getFeatureType() == " long":
                if self.architecture == "intel":
                    code = indent+"\"mov "+str(head.feature*8)+"(%1), %%rbx;\"\n"
                else:
                    code = indent+"\"ldrs x1, [%1, "+str(head.feature*8)+"];\"\n"
            elif self.getFeatureType() == "unsigned long":
                if self.architecture == "intel":
                    code = indent+"\"mov "+str(head.feature*8)+"(%1), %%rbx;\"\n"
                else:
                    code = indent+"\"ldr x1, [%1, "+str(head.feature*8)+"];\"\n"

            preswap=False

            code += indent+"//="+str(head.split)+"\n"
            if self.getFeatureType() == "float":
                if self.architecture == "intel":
                    if head.split < 0:
                        code += indent+"\"mov $"+float_to_bin(-1*head.split)+", %%eax;\"\n"
                    else:
                        code += indent+"\"mov $"+float_to_bin(head.split)+", %%eax;\"\n"
                    if head.split < 0:
                        code += indent+"\"mov $0x80000000, %%ecx;\"\n"
                        code += indent+"\"xor %%ecx, %%ebx;\"\n"
                        preswap=True
                    code += indent+"\"cmp %%ebx, %%eax;\"\n"
                else:
                    bin_string=float_to_bin(head.split)
                    if head.split < 0:
                        bin_string=float_to_bin(-1*head.split)
                    code += indent+"//\"mov x2, #"+float_to_bin(head.split)+";\"\n"
                    code += indent+"\"movz x2, #0x"+bin_string[-4:]+";\"\n"
                    code += indent+"\"movk x2, #0x"+bin_string[-8:-4]+", lsl 16;\"\n"
                    if head.split < 0:
                        code += indent+"\"movz x3, #0x8000, lsl 16;\"\n"
                        code += indent+"\"eor x1, x1, x3;\"\n"
                        preswap=True
                    # code += indent+"\"sxtw x1, w1;\"\n"
                    #Do actual comparison
                    code += indent+"\"cmp w1, w2;\"\n"
            elif self.getFeatureType() == "double":
                if self.architecture == "intel":
                    if head.split < 0:
                        code += indent+"\"mov $"+double_to_bin(-1*head.split)+", %%rax;\"\n"
                    else:
                        code += indent+"\"mov $"+double_to_bin(head.split)+", %%rax;\"\n"

                    if head.split < 0:
                        code += indent+"\"mov %%rcx, $0x8000000000000000;\"\n"
                        code += indent+"\"xor %%rcx, %%rbx;\"\n"
                        preswap=True
                        # code += indent+"\"mov %%eax, %%ecx;\"\n"
                        # code += indent+"\"and %%ebx, %%ecx;\"\n"
                        # code += indent+"\"shr $31, %%ecx;\"\n"
                        # code += indent+"\"sub %%ecx, %%ebx;\"\n"
                    code += indent+"\"cmp %%rbx, %%rax;\"\n"
                else:
                    bin_string=double_to_bin(head.split)
                    code += indent+"//\"mov x2, #"+double_to_bin(head.split)+";\"\n"
                    code += indent+"\"movz x2, #0x"+bin_string[-4:]+";\"\n"
                    code += indent+"\"movk x2, #0x"+bin_string[-8:-4]+", lsl 16;\"\n"
                    code += indent+"\"movk x2, #0x"+bin_string[-12:-8]+", lsl 32;\"\n"
                    code += indent+"\"movk x2, #0x"+bin_string[-16:-12]+", lsl 48;\"\n"
                    if head.split < 0:
                        code += indent+"\"movz x3, #0x8000, lsl 48;\"\n"
                        code += indent+"\"eor x1, x1, x3;\"\n"
                        preswap=True
                    #Do actual comparison
                    code += indent+"\"cmp x1, x2;\"\n"
            else:
                if self.architecture == "intel":
                    code += indent+"\"mov $"+str(int(head.split))+", %%rax;\"\n"
                    #Do actual comparison
                    code += indent+"\"cmp %%rbx, %%rax;\"\n"
                else:
                    bin_string=long_to_bin(head.split)
                    code += indent+"//\"mov x2, #"+long_to_bin(head.split)+";\"\n"
                    code += indent+"\"movz x2, #0x"+bin_string[-4:]+";\"\n"
                    code += indent+"\"movk x2, #0x"+bin_string[-8:-4]+", lsl 16;\"\n"
                    code += indent+"\"movk x2, #0x"+bin_string[-12:-8]+", lsl 32;\"\n"
                    code += indent+"\"movk x2, #0x"+bin_string[-16:-12]+", lsl 48;\"\n"
                    #Do actual comparison
                    code += indent+"\"cmp x1, x2;\"\n"

            swap=head.probRight > head.probLeft
            label=label_idx[0]
            label_idx[0]+=1
            condswap=(swap!=preswap)
            if self.architecture == "intel":
                code += indent+"#ifdef SPLIT_DATATYPE_UNSIGNED\n"
                # code += indent+"\"j"+("n" if swap else "")+"b __rtitt_lab_"+str(label)+"_"+str(treeID)+";\"\n"
                code += indent+"\"j"+("n" if condswap else "")+"l __rtitt_lab_"+str(label)+"_"+str(treeID)+";\"\n"
                code += indent+"#else\n"
                code += indent+"\"j"+("n" if condswap else "")+"l __rtitt_lab_"+str(label)+"_"+str(treeID)+";\"\n"
                code += indent+"#endif\n"
            else:
                code += indent+"\"b."+("le" if condswap else "gt")+" __rtitt_lab_"+str(label)+"_"+str(treeID)+";\"\n"

            code += write_tree((head.rightChild if swap else head.leftChild), indent+"   ")

            #Define else branch
            code += indent+"\"__rtitt_lab_"+str(label)+"_"+str(treeID)+":\"\n"
            code += write_tree((head.leftChild if swap else head.rightChild), indent+"   ")
            return code

        ifelsecode=write_tree(head)

        cppCode = """
                    inline unsigned int {namespace}_predict{treeID}({feature_t} const pX[{dim}]){
                            volatile unsigned int result;
                            float splittarget=0;
                            asm volatile(
                            \n{ifelsecode}\n
                            "__rtitt_end_{treeID}:"
                            """.replace("{treeID}", str(treeID)) \
                                .replace("{dim}", str(self.dim)) \
                                .replace("{ifelsecode}", ifelsecode) \
                                .replace("{namespace}", self.namespace) \
                                .replace("{feature_t}", self.getFeatureType())

        if self.architecture == "intel":
            cppCode += """:"=r"(result) :"r"(pX),"r"(&splittarget):"xmm1", "xmm2","xmm3","rax","rbx","rcx");"""
        else:
            cppCode += """:"=r"(result) :"r"(pX),"r"(&splittarget):"d1", "d2","d3","x1","x2","x3");"""

        cppCode += """
                            return result;
                        }
                    """

        return cppCode, 0

class IntLogicStandardIFTreeConverter(TreeConverter):
    """ A IfTreeConverter converts a DecisionTree into its if-else structure in c language
    """
    def __init__(self, dim, namespace, featureType):
        super().__init__(dim, namespace, featureType)

    def getImplementation(self, treeID, head, level = 1):
        """ Generate the actual if-else implementation for a given node

        Args:
            treeID (TYPE): The id of this tree (in case we are dealing with a forest)
            head (TYPE): The current node to generate an if-else structure for.
            level (int, optional): The intendation level of the generated code for easier
                                                        reading of the generated code

        Returns:
            String: The actual if-else code as a string
        """
        featureType = self.getFeatureType()
        intLogicType="int" if featureType == "float" else "long"
        # headerCode = "inline float {namespace}Forest_predict{treeID}({feature_t} const pX[{dim}], float pred[{numClasses}]);\n" \
        #                                 .replace("{treeID}", str(treeID)) \
        #                                 .replace("{dim}", str(self.dim)) \
        #                                 .replace("{namespace}", self.namespace) \
        #                                 .replace("{feature_t}", featureType)
        code = ""
        tabs = "".join(['\t' for i in range(level)])

        if head.prediction is not None:
            # for i in range(len(head.prediction)):
            #     code += tabs + "pred[" + str(i) + "] += " + str(head.prediction[i]) + ";\n"

            return tabs + "return " + str(int(np.argmax(head.prediction))) + ";\n" ;
            #return tabs + "return " + str(int(head.prediction)) + ";\n" ;
            #return tabs + "return " + str(float(head.prediction)) + ";\n" ;
        else:
                if featureType != "float" and featureType != "double":
                    code += tabs + "if("+"pX[" + str(head.feature)+"] <= (float) " + ("{:f}".format(head.split))+"){\n"
                else:
                    negatstring=""
                    splitval=head.split
                    op="<="
                    if splitval < 0:
                        splitval=-1*splitval
                        negatstring="^ (0b1 << "+("31" if featureType=="float" else "63")+")"
                        op=">"
                    code += tabs + "if((*( (("+intLogicType+" *)(pX)) + "+str(head.feature)+" )"+ negatstring +")"+ op + "(("+intLogicType+")("+(float_to_bin(splitval) if featureType == "float" else double_to_bin(splitval))+"))" + "){\n"
                code += self.getImplementation(treeID, head.leftChild, level + 1)
                code += tabs + "} else {\n"
                code += self.getImplementation(treeID, head.rightChild, level + 1)
                code += tabs + "}\n"

        return code

    def getCode(self, tree, treeID, numClasses):
        """ Generate the actual if-else implementation for a given tree

        Args:
            tree (TYPE): The tree
            treeID (TYPE): The id of this tree (in case we are dealing with a forest)

        Returns:
            Tuple: A tuple (headerCode, cppCode), where headerCode contains the code (=string) for
            a *.h file and cppCode contains the code (=string) for a *.cpp file
        """
        featureType = self.getFeatureType()
        cppCode = "inline unsigned int {namespace}_predict{treeID}({feature_t} const pX[{dim}]){\n" \
                                .replace("{treeID}", str(treeID)) \
                                .replace("{dim}", str(self.dim)) \
                                .replace("{namespace}", self.namespace) \
                                .replace("{feature_t}", featureType) \
                                .replace("{numClasses}", str(numClasses))

        cppCode += self.getImplementation(treeID, tree.head)
        cppCode += "}\n"

        headerCode = "inline unsigned int {namespace}_predict{treeID}({feature_t} const pX[{dim}]);\n" \
                                        .replace("{treeID}", str(treeID)) \
                                        .replace("{dim}", str(self.dim)) \
                                        .replace("{namespace}", self.namespace) \
                                        .replace("{feature_t}", featureType) \
                                        .replace("{numClasses}", str(numClasses))


        return headerCode, cppCode