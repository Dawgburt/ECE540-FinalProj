/*
* SPDX-License-Identifier: Apache-2.0
* Copyright 2019 Western Digital Corporation or its affiliates.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http:*www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/
/**
* @file   demo_platform_al.h
* @author Nati Rapaport
* @date   23.10.2019
* @brief  initializations and api layer to the specific platform functions 
*/
#ifndef  __DEMO_PLATFORM_AL_H__
#define  __DEMO_PLATFORM_AL_H__

/**
* include files
*/
#include "psp_api.h"

#ifdef D_HI_FIVE1
   #include "encoding.h"
   #include "platform.h"
#endif

/**
* definitions
*/

/* LEDs outputs definition */
#ifdef D_HI_FIVE1
    #define D_LED_GREEN_ON 0
    #define D_LED_BLUE_ON  1
#elif defined(D_SWERV_EH1) || defined(D_SWERV_EH2) || defined(D_SWERV_EL2)
    #define D_LED_GREEN_ON 'X' //unknown
    #define D_LED_BLUE_ON  'X' //unknown
#else
    #define D_LED_GREEN_ON 'X' //unknown
    #define D_LED_BLUE_ON  'X' //unknown
   PRE_COMPILED_MSG("no platform was defined")
#endif

/**
* macros
*/

/**
* types
*/

/**
* local prototypes
*/

/**
* external prototypes
*/
#ifdef D_HI_FIVE1
   extern void _init();
#endif


/**
* global variables
*/

/**
* global variables
*/

/**
* functions
*/

/**
* demoPlatformInit - Initialize board related stuff
*
* */
void demoPlatformInit(void);


/**
* @brief demoOutputMsg - output (usually, but not necessarily - print it out) a given string,
*                 using the platform means for that.
*
* const void *pStr - pointer to a string to be printed out
*
* u32_t uiSize - number of characters to print
*
* */
#ifdef D_HI_FIVE1
   void demoOutputMsg(const void *pStr, u32_t uiSize);
#elif defined(D_SWERV_EH1) || defined(D_SWERV_EH2) || defined(D_SWERV_EL2)
   #include "bsp_printf.h"
   #define demoOutputMsg(f_,...)  ee_printf((f_), ##__VA_ARGS__)
#else
   #define demoOutputMsg(f_,...)
#endif

/**
* @brief demoOutputToggelLed - sets LED output according input request.
*
*
* The "LED action" is defined per each platform
*
* */
void demoOutputToggelLed(void);

/**
* @brief demoOutputLed - sets LED output on/off
*
* const uiOnOffMode = 0/1
*
*
* */
void demoOutpuLed(const u08_t ucOnOffMode);


/**
*
* @brief Indicates whether the running target is Swerv board or Whisper Instruction-set simulator
*
* @return 1 if Swerv board , 0 otherwise
*/
u32_t demoIsSwervBoard(void);

#endif /* __DEMO_PLATFORM_AL_H__ */
