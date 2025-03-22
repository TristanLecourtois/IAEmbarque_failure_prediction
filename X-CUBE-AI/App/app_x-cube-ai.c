/**
  ******************************************************************************
  * @file    app_x-cube-ai.c
  * @author  X-CUBE-AI
  * @brief   Exemple d'implémentation de code d'inférence (6 entrées, 5 classes),
  *          avec un réseau nommé "network".
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2025 STMicroelectronics.
  * All rights reserved.
  *
  * Ce logiciel est régi par les termes disponibles dans le fichier LICENSE,
  * fourni dans le répertoire racine de ce logiciel.
  *
  ******************************************************************************
  */

#ifdef __cplusplus
 extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <inttypes.h>
#include <string.h>

#include "app_x-cube-ai.h"
#include "main.h"
#include "ai_datatypes_defines.h"

/* -- Fichiers générés par le pack X-CUBE-AI pour votre réseau nommé "network" */
#include "network.h"
#include "network_data.h"

/* --- USER CODE BEGIN Includes --- */
extern UART_HandleTypeDef huart2;

/* Paramètres : 6 floats en entrée, 5 classes en sortie */
#define BYTES_IN_FLOATS   (6 * 4)   // 6 floats => 6*4 = 24 octets
#define CLASS_NUMBER      (5)       // 5 classes
#define TIMEOUT           (1000)

#define SYNCHRONISATION   0xAB
#define ACKNOWLEDGE       0xCD

/* --- USER CODE END Includes --- */


/* IO buffers ----------------------------------------------------------------*/
#if !defined(AI_NETWORK_INPUTS_IN_ACTIVATIONS)
AI_ALIGNED(4) ai_i8 data_in_1[AI_NETWORK_IN_1_SIZE_BYTES];
ai_i8* data_ins[AI_NETWORK_IN_NUM] = {
    data_in_1
};
#else
ai_i8* data_ins[AI_NETWORK_IN_NUM] = {
    NULL
};
#endif

#if !defined(AI_NETWORK_OUTPUTS_IN_ACTIVATIONS)
AI_ALIGNED(4) ai_i8 data_out_1[AI_NETWORK_OUT_1_SIZE_BYTES];
ai_i8* data_outs[AI_NETWORK_OUT_NUM] = {
    data_out_1
};
#else
ai_i8* data_outs[AI_NETWORK_OUT_NUM] = {
    NULL
};
#endif

/* Activations buffer --------------------------------------------------------*/
AI_ALIGNED(32)
static uint8_t pool0[AI_NETWORK_DATA_ACTIVATION_1_SIZE];

ai_handle data_activations0[] = { pool0 };

/* AI objects ----------------------------------------------------------------*/
static ai_handle network = AI_HANDLE_NULL;  // Handle principal de notre réseau
static ai_buffer* ai_input  = NULL;
static ai_buffer* ai_output = NULL;

/* ----------------------------------------------------------------------------
 *  Fonction interne d’affichage d’erreur
 * ---------------------------------------------------------------------------*/
static void ai_log_err(const ai_error err, const char *fct)
{
  if (fct)
    printf("AI Error (%s) - type=0x%02x code=0x%02x\r\n", fct,
           err.type, err.code);
  else
    printf("AI Error - type=0x%02x code=0x%02x\r\n", err.type, err.code);

  // Boucle infinie pour signaler l'erreur
  while (1) { ; }
}

/* ----------------------------------------------------------------------------
 *  Initialisation du réseau de neurones
 * ---------------------------------------------------------------------------*/
static int ai_boostrap(ai_handle *act_addr)
{
  ai_error err;

  /* Création et initialisation du réseau "network" */
  err = ai_network_create_and_init(&network, act_addr, NULL);
  if (err.type != AI_ERROR_NONE) {
    ai_log_err(err, "ai_network_create_and_init");
    return -1;
  }

  /* Récupère les buffers d'entrées et de sorties */
  ai_input  = ai_network_inputs_get(network, NULL);
  ai_output = ai_network_outputs_get(network, NULL);

#if defined(AI_NETWORK_INPUTS_IN_ACTIVATIONS)
  /* Si l'option "--allocate-inputs" est activée,
   * le buffer d'entrée est déjà alloué dans la zone d'activation.
   */
  for (int idx = 0; idx < AI_NETWORK_IN_NUM; idx++) {
    data_ins[idx] = ai_input[idx].data;
  }
#else
  /* Sinon on pointe data_ins[idx] vers le buffer data_in_1 (statiques) */
  for (int idx = 0; idx < AI_NETWORK_IN_NUM; idx++) {
    ai_input[idx].data = data_ins[idx];
  }
#endif

#if defined(AI_NETWORK_OUTPUTS_IN_ACTIVATIONS)
  /* Idem pour la sortie si "--allocate-outputs" est activée. */
  for (int idx = 0; idx < AI_NETWORK_OUT_NUM; idx++) {
    data_outs[idx] = ai_output[idx].data;
  }
#else
  for (int idx = 0; idx < AI_NETWORK_OUT_NUM; idx++) {
    ai_output[idx].data = data_outs[idx];
  }
#endif

  return 0;
}

/* ----------------------------------------------------------------------------
 *  Exécution de l’inférence
 * ---------------------------------------------------------------------------*/
static int ai_run(void)
{
  ai_i32 batch;

  /* On lance le réseau, 1 exécution */
  batch = ai_network_run(network, ai_input, ai_output);
  if (batch != 1) {
    ai_log_err(ai_network_get_error(network), "ai_network_run");
    return -1;
  }

  return 0;
}

/* --- USER CODE BEGIN 2 ---------------------------------------------------- */
/**
  * @brief  Synchronisation via UART (en attente de SYNCHRONISATION, puis envoi d’un ACK).
  */
void synchronize_UART(void)
{
  bool is_synced = false;
  unsigned char rx[2] = {0};
  unsigned char tx[2] = {ACKNOWLEDGE, 0};

  while (!is_synced)
  {
    // On reçoit 2 octets (même si on n'en lit réellement qu'1)
    HAL_UART_Receive(&huart2, (uint8_t *)rx, sizeof(rx), TIMEOUT);
    if (rx[0] == SYNCHRONISATION)
    {
      HAL_UART_Transmit(&huart2, (uint8_t *)tx, sizeof(tx), TIMEOUT);
      is_synced = true;
    }
  }
}

/**
  * @brief  Reçoit 6 floats depuis l'UART (24 octets) puis les place en entrée du réseau.
  * @param  data[] : pointeur vers le buffer d'entrée du NN
  * @retval 0 si OK, 1 si erreur
  */
int acquire_and_process_data(ai_i8 *data[])
{
  unsigned char tmp[BYTES_IN_FLOATS] = {0}; // 24 octets
  int num_elements = sizeof(tmp);           // 24
  int num_floats   = num_elements / 4;      // 6

  // 1) Réception de 24 octets via UART
  HAL_StatusTypeDef status = HAL_UART_Receive(&huart2, (uint8_t*)tmp, num_elements, TIMEOUT);
  if (status != HAL_OK)
  {
    printf("Erreur de réception UART (acquire_and_process_data). Code: %d\r\n", status);
    return 1;
  }

  // 2) Vérification
  if (num_elements % 4 != 0)
  {
    printf("Taille non multiple de 4 => impossible de reconstruire 6 floats.\r\n");
    return 1;
  }

  // 3) Reconstruction des floats (6 * 4 octets) dans le buffer d'entrée
  for (size_t i = 0; i < (size_t)num_floats; i++)
  {
    unsigned char bytes[4];
    for (size_t j = 0; j < 4; j++)
    {
      bytes[j] = tmp[i * 4 + j];
    }
    // Copie dans data (cast en (uint8_t*) pour le buffer final)
    for (size_t k = 0; k < 4; k++)
    {
      ((uint8_t *)data)[(i * 4 + k)] = bytes[k];
    }
  }

  return 0; // OK
}

/**
  * @brief  Post-traitement : lit 5 probabilités en sortie du NN et les envoie via UART.
  * @param  data[] : pointeur vers le buffer de sortie (5 floats)
  * @retval 0 si OK, 1 si erreur
  */
int post_process(ai_i8 *data[])
{
  if (data == NULL)
  {
    printf("La sortie du réseau est NULL.\r\n");
    return 1;
  }

  uint8_t *output = (uint8_t*)data;
  float    outs[CLASS_NUMBER]      = {0.0f};
  uint8_t  outs_uint8[CLASS_NUMBER] = {0};

  // Lecture des 5 floats (chaque float = 4 octets)
  for (size_t i = 0; i < CLASS_NUMBER; i++)
  {
    uint8_t temp[4] = {0};
    for (size_t j = 0; j < 4; j++)
    {
      temp[j] = output[i * 4 + j];
    }
    // Recompose le float
    outs[i] = *(float*)&temp;
    // Convertit en [0..255]
    outs_uint8[i] = (uint8_t)(outs[i] * 255.0f);
  }

  // Envoi UART : on envoie 5 octets (les 5 probas)
  HAL_StatusTypeDef status = HAL_UART_Transmit(&huart2, (uint8_t*)outs_uint8, CLASS_NUMBER, TIMEOUT);
  if (status != HAL_OK)
  {
    printf("Erreur transmission UART (post_process). Code: %d\r\n", status);
    return 1;
  }

  return 0; // OK
}
/* --- USER CODE END 2 ------------------------------------------------------ */

/* ----------------------------------------------------------------------------
 *  Fonctions appelées par le main
 * ---------------------------------------------------------------------------*/

/**
  * @brief  Initialise le réseau de neurones (appellée 1 fois au démarrage)
  */
void MX_X_CUBE_AI_Init(void)
{
  printf("\r\n=== Initialisation du réseau <network> ===\r\n");
  if (ai_boostrap(data_activations0) != 0)
  {
    printf("Echec initialisation réseau.\r\n");
  }
}

/**
  * @brief  Exécute en boucle : synchronisation, acquisition, inference, post-traitement
  */
void MX_X_CUBE_AI_Process(void)
{
  int res = -1;

  // Pointeurs vers buffers d'entrée / sortie
  uint8_t *in_data  = (uint8_t*)ai_input[0].data;
  uint8_t *out_data = (uint8_t*)ai_output[0].data;

  // 1) Synchronisation UART
  synchronize_UART();

  // 2) Boucle d'inférence continue
  while (1)
  {
    // a) Acquisition
    res = acquire_and_process_data((ai_i8**)in_data);
    if (res != 0) break;

    // b) Exécution du réseau
    res = ai_run();
    if (res != 0) break;

    // c) Post-traitement
    res = post_process((ai_i8**)out_data);
    if (res != 0) break;
  }

  // Si on arrive ici, il y a eu une erreur => affichage puis boucle infinie
  if (res != 0)
  {
    ai_error err = { AI_ERROR_INVALID_STATE, AI_ERROR_CODE_NETWORK };
    ai_log_err(err, "Process a échoué");
  }
}

#ifdef __cplusplus
}
#endif
