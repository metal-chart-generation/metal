import os
import json
import argparse
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import multiprocessing

## Metal agents
from agents.GenerationAgent import GenerationAgent
from agents.TextCritiqueAgent import TextCritiqueAgent
from agents.VisualCritiqueAgent import VisualCritiqueAgent
from agents.RevisionAgent import RevisionAgent
from agents.VerificationAgent import VerificationAgent

from agents.EvaluationAgent import EvaluationAgent

## Variants
from agents.SingleCritiqueAgent import SingleCritiqueAgent
from agents.TextCritiqueVisualAgent import TextCritiqueVisualAgent
from agents.VisualRevisionAgent import VisualRevisionAgent

import torch
import gc

# utils
def setup_logger(output_path, output_suffix):
    logger = logging.getLogger(output_suffix)
    logger.setLevel(logging.INFO)

    log_file = os.path.join(output_path, f"{output_suffix}.log")
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger

def format_log_critique(message):
    msg = "             >> =======================================================\n"
    for line in message.split("\n"):
        msg += f"                                               {line}\n"
    msg += "                                             >> ======================================================="
    return msg

# 1. Metal: GenerationAgent, TextCritiqueAgent, VisualCritiqueAgent, RevisionAgent, VerificationAgent, EvaluationAgent
def process_data(data_item, config, model, max_iterations):
    log = []
    
    output_suffix = os.path.splitext(os.path.basename(data_item["GroundTruthFigure"]))[0]
    output_prefix = "direct_generation"
    output_path = f"{config['working_dir']}{output_suffix}/"
    os.makedirs(output_path, exist_ok=True)
    
    logger = setup_logger(output_path, output_suffix)
    
    try:
        logger.info(f"## Processing data {output_suffix} ##")
        
        ground_truth_img_path = os.path.join(config["img_parent_dir"], data_item["GroundTruthFigure"])
        ground_truth_code_path = os.path.join(config["img_parent_dir"], data_item["GroundTruthFigureCode"])
        generation_prompt = data_item["Instruction"]
        
        agents = {
            "GenerationAgent": GenerationAgent(model, **config.get('agent_kwargs', {})),
            "TextCritiqueAgent": TextCritiqueAgent(model, **config.get('agent_kwargs', {})),
            "VisualCritiqueAgent": VisualCritiqueAgent(model, **config.get('agent_kwargs', {})),
            "RevisionAgent": RevisionAgent(model, **config.get('agent_kwargs', {})),
            "EvaluationAgent": EvaluationAgent(model, **config.get('agent_kwargs', {})),
            "VerificationAgent": VerificationAgent(model, **config.get('agent_kwargs', {})),
        }
        generation_agent = agents["GenerationAgent"]
        text_critique_agent = agents["TextCritiqueAgent"]
        visual_critique_agent = agents["VisualCritiqueAgent"]
        revision_agent = agents["RevisionAgent"]
        verification_agent = agents["VerificationAgent"]
        evaluation_agent = agents["EvaluationAgent"]
        
        
        """
        Direct Generation
        """
        logger.info("  |-- Direct Generation")
       
        raw_response, generated_code, generated_code_path, generated_image_path, combined_image_path = generation_agent.act(
            generation_prompt, ground_truth_img_path, output_path, output_prefix, output_suffix)
        gc.collect()
        torch.cuda.empty_cache()
        
        logger.info("  |-- Direct Evaluation")
        generated_code_evaluate_path = f"{os.path.splitext(generated_code_path)[0]}_evaluate.py"
        evaluation_results, evaluate_revision_rank = evaluation_agent.act(
            ground_truth_code_path, generated_code_evaluate_path)
        log_entry = {
            "iteration": 0,
            "agent": "GenerationAgent",
            "generation_prompt": generation_prompt,
            "raw_response": raw_response,
            "generated_code": generated_code,
            "generated_code_path": generated_code_path,
            "combined_image_path": combined_image_path,
            "evaluation_results": evaluation_results,
        }
        log.append(log_entry)
        for metric, value in evaluation_results.items():
            logger.info(f"      >>> {metric}: {value}")
            
        logger.info("  |-- Direct Verification")
        verification_results, revision_rank = verification_agent.act(
            ground_truth_img_path, generated_image_path)
        log.append({
            "iteration": 0,
            "agent": "VerificationAgent",
            "verification_results": verification_results,
        })
        for metric, value in verification_results.items():
            logger.info(f"      >>> {metric}: {value}")
        
        if not revision_rank or max_iterations == -1:
            with open(f"{output_path}{output_suffix}_log.json", "w") as f:
                        json.dump(log, f, indent=2)
            return {"data": output_suffix, "log": log, "flag": "perfect"}
        
        error_file_path = f"{os.path.splitext(generated_code_path)[0]}_error.txt"
        if os.path.exists(error_file_path):
            with open(error_file_path) as f:
                error = f.read()
            logger.warning(f"Detected error: {error}. Try Again.")
            os.system(f"rm {error_file_path}")
            log = []
            
            logger.info("  |-- Direct Generation (Try Again)")
       
            raw_response, generated_code, generated_code_path, generated_image_path, combined_image_path = generation_agent.act(
                generation_prompt, ground_truth_img_path, output_path, output_prefix, output_suffix)
            gc.collect()
            torch.cuda.empty_cache()
            
            logger.info("  |-- Direct Evaluation (Try Again)")
            generated_code_evaluate_path = f"{os.path.splitext(generated_code_path)[0]}_evaluate.py"
            evaluation_results, evaluate_revision_rank = evaluation_agent.act(
                ground_truth_code_path, generated_code_evaluate_path)
            log_entry = {
                "iteration": 0,
                "agent": "GenerationAgent",
                "generation_prompt": generation_prompt,
                "raw_response": raw_response,
                "generated_code": generated_code,
                "generated_code_path": generated_code_path,
                "combined_image_path": combined_image_path,
                "evaluation_results": evaluation_results,
            }
            log.append(log_entry)
            for metric, value in evaluation_results.items():
                logger.info(f"      >>> {metric}: {value}")
            
            logger.info("  |-- Direct Verification (Try Again)")
            verification_results, revision_rank = verification_agent.act(
                ground_truth_img_path, generated_image_path)
            log.append({
                "iteration": 0,
                "agent": "VerificationAgent",
                "verification_results": verification_results,
            })
            for metric, value in verification_results.items():
                logger.info(f"      >>> {metric}: {value}")
            
            if not revision_rank or max_iterations == -1:
                with open(f"{output_path}{output_suffix}_log.json", "w") as f:
                            json.dump(log, f, indent=2)
                return {"data": output_suffix, "log": log, "flag": "perfect"}
            
            error_file_path = f"{os.path.splitext(generated_code_path)[0]}_error.txt"
            if os.path.exists(error_file_path):
                with open(error_file_path) as f:
                    error = f.read()
            
                log.append({
                    "iteration": 0,
                    "agent": "Detected Error, Skip Iterative Critique and Revision",
                    "error": error,
                })
                logger.warning(f"Detected error: {error}. Skipping iterative critique and revision.")
                return {"data": output_suffix, "log": log, "flag": "error"}
        
        
        """
        Iterative Critique and Revision
        """
        improved_flg = False
        avg_improvement = 0
        for j in range(1, max_iterations + 1):
            if not revision_rank or len(revision_rank) == 0:
                with open(f"{output_path}{output_suffix}_log.json", "w") as f:
                        json.dump(log, f, indent=2)
                break
            
            logger.info(f"   |--- Iteration {j}/{max_iterations}")
            lowest_metric = revision_rank[0]
            try:
                logger.info(f"    |---- [iteration {j}] Revising Metric: {lowest_metric} in {', '.join(revision_rank)}")
                output_prefix = f"revised_round_{j}_{lowest_metric}"
                
                """
                Visual Critique
                """
                logger.info(f"     |----- [iteration {j}] Visual Critique")
                visual_prompt, visual_res = visual_critique_agent.act(
                    lowest_metric, combined_image_path)
                log.append({
                    "iteration": j,
                    "agent": "VisualCritiqueAgent",
                    "lowest_metric": lowest_metric,
                    "visual_critique_prompt": visual_prompt,
                    "visual_critique_res": visual_res,
                })
                logger.info(format_log_critique(visual_res))
                gc.collect()
                torch.cuda.empty_cache()
                
                """
                Text Critique
                """
                logger.info(f"     |----- [iteration {j}] Code Critique")
                text_prompt, text_res = text_critique_agent.act(
                    visual_res, generated_code)
                log.append({
                    "iteration": j,
                    "agent": "TextCritiqueAgent",
                    "lowest_metric": lowest_metric,
                    "text_critique_prompt": text_prompt,
                    "text_critique_res": text_res,
                })
                logger.info(format_log_critique(text_res))
                gc.collect()
                torch.cuda.empty_cache()
                                    
                """
                Revision
                """
                logger.info(f"     |----- [iteration {j}] Revision")
                revision_prompt, revision_raw_response, revised_code, revised_code_path, revised_img_path, revised_pdf_path, revised_combined_image_path = revision_agent.act(
                    ground_truth_img_path, text_res, output_path, output_prefix, output_suffix)
                
                logger.info(f"     |----- [iteration {j}] Revised Evaluation")
                
                revised_code_evaluate_path = f"{os.path.splitext(revised_code_path)[0]}_evaluate.py"
                new_evaluation_results, new_evaluation_revision_rank = evaluation_agent.act(
                    ground_truth_code_path, revised_code_evaluate_path)
                
                for metric, value in new_evaluation_results.items():
                    old_value = evaluation_results.get(metric, 0)
                    changed = value - old_value
                    logger.info(f"            >>> {metric}: {value} ({'+' if changed >=0 else ''}{changed})")
                    
                evaluation_improved = True if new_evaluation_results.get(lowest_metric, 0) > evaluation_results.get(lowest_metric, 0) else False
                logger.info(f"            >>> evaluation improved: {evaluation_improved}")
                
                
                logger.info(f"     |----- [iteration {j}] Revised Verification")
                new_verification_results, new_revision_rank = verification_agent.act(
                    ground_truth_img_path, revised_img_path)
                
                for metric, value in new_verification_results.items():
                    old_value = verification_results.get(metric, 0)
                    changed = value - old_value
                    logger.info(f"            >>> {metric}: {value} ({'+' if changed >=0 else ''}{changed})")
                
                
                improved = True if new_verification_results.get(lowest_metric, 0) > verification_results.get(lowest_metric, 0) else False
                logger.info(f"            >>> verification improved: {improved}")
                
                log_entry = {
                    "iteration": j,
                    "agent": "RevisionAgent",
                    "revision_prompt": revision_prompt,
                    "revision_raw_response": revision_raw_response,
                    "revised_code": revised_code,
                    "revised_code_path": revised_code_path,
                    "revised_img_path": revised_img_path,
                    "revised_pdf_path": revised_pdf_path,
                    "revised_combined_image_path": revised_combined_image_path,
                    "new_evaluation_results": new_evaluation_results,
                    "new_verification_results": new_verification_results,
                    "improved": improved,
                    "evaluation_improved": evaluation_improved,
                }
                log.append(log_entry)
                gc.collect()
                torch.cuda.empty_cache()
            
                if improved:
                    verification_results = new_verification_results
                    revision_rank = new_revision_rank
                    generated_code = revised_code
                    generated_code_path = revised_code_path
                    combined_image_path = revised_combined_image_path
                    
                    old_avg = sum(evaluation_results.values()) / len(evaluation_results)
                    new_avg = sum(new_evaluation_results.values()) / len(new_evaluation_results)
                    improvement = new_avg - old_avg
                    if improvement > 0:
                        improved_flg = True
                        avg_improvement = max(improvement, avg_improvement)
                        evaluation_results = new_evaluation_results
                    
            except Exception as e:
                logger.error(f"    ** Error: {e} **", exc_info=True)
                log.append({
                    "iteration": j,
                    "agent": "Error",
                    "error": str(e),
                })
                continue
            finally:
                with open(f"{output_path}{output_suffix}_log.json", "w") as f:
                    json.dump(log, f, indent=2)
        
        return {"data": output_suffix, "log": log, "flag": "improved" if improved_flg else "not_improved", "improvement": avg_improvement}
    
    except Exception as e:
        logger.error(f"## Failed to process data {output_suffix}: {e} ##", exc_info=True)
        log.append({
            "data": output_suffix,
            "agent": "Error",
            "error": str(e),
        })
        return {"data": output_suffix, "log": log, "flag": "error", "improvement": 0}

# 2. Metal-s: GenerationAgent, SingleCritiqueAgent, RevisionAgent, VerificationAgent, EvaluationAgent
def sinlge_critique_process_data(data_item, config, model, max_iterations):
    log = []
    
    output_suffix = os.path.splitext(os.path.basename(data_item["GroundTruthFigure"]))[0]
    output_prefix = "direct_generation"
    output_path = f"{config['working_dir']}{output_suffix}/"
    os.makedirs(output_path, exist_ok=True)
    
    logger = setup_logger(output_path, output_suffix)
    
    try:
        logger.info(f"## Processing data {output_suffix} ##")
        
        ground_truth_img_path = os.path.join(config["img_parent_dir"], data_item["GroundTruthFigure"])
        ground_truth_code_path = os.path.join(config["img_parent_dir"], data_item["GroundTruthFigureCode"])
        generation_prompt = data_item["Instruction"]
        
        agents = {
            "GenerationAgent": GenerationAgent(model, **config.get('agent_kwargs', {})),
            "SingleCritiqueAgent": SingleCritiqueAgent(model, **config.get('agent_kwargs', {})),
            "RevisionAgent": RevisionAgent(model, **config.get('agent_kwargs', {})),
            "EvaluationAgent": EvaluationAgent(model, **config.get('agent_kwargs', {})),
            "VerificationAgent": VerificationAgent(model, **config.get('agent_kwargs', {})),
        }
        generation_agent = agents["GenerationAgent"]
        single_critique_agent = agents["SingleCritiqueAgent"]
        revision_agent = agents["RevisionAgent"]
        verification_agent = agents["VerificationAgent"]
        evaluation_agent = agents["EvaluationAgent"]
        
        """
        Direct Generation
        """
        logger.info("  |-- Direct Generation")
       
        raw_response, generated_code, generated_code_path,  generated_image_path, combined_image_path = generation_agent.act(
            generation_prompt, ground_truth_img_path, output_path, output_prefix, output_suffix)
        gc.collect()
        torch.cuda.empty_cache()
        
        generated_code_evaluate_path = f"{os.path.splitext(generated_code_path)[0]}_evaluate.py"
        evaluation_agent = agents["EvaluationAgent"]
        logger.info("  |-- Direct Evaluation")
        
        evaluation_results, revision_rank = evaluation_agent.act(
            ground_truth_code_path, generated_code_evaluate_path)
        log_entry = {
            "iteration": 0,
            "agent": "GenerationAgent",
            "generation_prompt": generation_prompt,
            "raw_response": raw_response,
            "generated_code": generated_code,
            "generated_code_path": generated_code_path,
            "combined_image_path": combined_image_path,
            "evaluation_results": evaluation_results,
        }
        log.append(log_entry)
        for metric, value in evaluation_results.items():
            logger.info(f"      >>> {metric}: {value}")
        
        if not revision_rank or max_iterations == -1:
            with open(f"{output_path}{output_suffix}_log.json", "w") as f:
                        json.dump(log, f, indent=2)
            return {"data": output_suffix, "log": log, "flag": "perfect"}
        
        error_file_path = f"{os.path.splitext(generated_code_path)[0]}_error.txt"
        if os.path.exists(error_file_path):
            with open(error_file_path) as f:
                error = f.read()
            logger.warning(f"Detected error: {error}. Try Again.")
            os.system(f"rm {error_file_path}")
            log = []
            
            logger.info("  |-- Direct Generation (Try Again)")
       
            raw_response, generated_code, generated_code_path, combined_image_path = generation_agent.act(
                generation_prompt, ground_truth_img_path, output_path, output_prefix, output_suffix)
            gc.collect()
            torch.cuda.empty_cache()
            
            generated_code_evaluate_path = f"{os.path.splitext(generated_code_path)[0]}_evaluate.py"
            evaluation_agent = agents["EvaluationAgent"]
            evaluation_results, revision_rank = evaluation_agent.act(
                ground_truth_code_path, generated_code_evaluate_path)
            logger.info("  |-- Direct Evaluation")
            log_entry = {
                "iteration": 0,
                "agent": "GenerationAgent",
                "generation_prompt": generation_prompt,
                "raw_response": raw_response,
                "generated_code": generated_code,
                "generated_code_path": generated_code_path,
                "combined_image_path": combined_image_path,
                "evaluation_results": evaluation_results,
            }
            log.append(log_entry)
            for metric, value in evaluation_results.items():
                logger.info(f"      >>> {metric}: {value}")
            
            if not revision_rank or max_iterations == -1:
                with open(f"{output_path}{output_suffix}_log.json", "w") as f:
                            json.dump(log, f, indent=2)
                return {"data": output_suffix, "log": log, "flag": "perfect"}
            
            error_file_path = f"{os.path.splitext(generated_code_path)[0]}_error.txt"
            if os.path.exists(error_file_path):
                with open(error_file_path) as f:
                    error = f.read()
            
                log.append({
                    "iteration": 0,
                    "agent": "Detected Error, Skip Iterative Critique and Revision",
                    "error": error,
                })
                logger.warning(f"Detected error: {error}. Skipping iterative critique and revision.")
                return {"data": output_suffix, "log": log, "flag": "error"}
        
        """
        Iterative Critique and Revision
        """
        improved_flg = False
        avg_improvement = 0
        
        for j in range(1, max_iterations + 1):
            if not revision_rank or len(revision_rank) == 0:
                with open(f"{output_path}{output_suffix}_log.json", "w") as f:
                        json.dump(log, f, indent=2)
                break
            
            logger.info(f"   |--- Iteration {j}/{max_iterations}")
            lowest_metric = revision_rank[0]
            try:
                logger.info(f"    |---- [iteration {j}] Revising Metric: {lowest_metric} in {', '.join(revision_rank)}")
                output_prefix = f"revised_round_{j}_{lowest_metric}"
                
                """
                Critique
                """
                logger.info(f"     |----- [iteration {j}] Critique")
                critique_prompt, critique_res = single_critique_agent.act(
                    lowest_metric, combined_image_path, generated_code)
                log.append({
                    "iteration": j,
                    "agent": "VisualCritiqueAgent",
                    "lowest_metric": lowest_metric,
                    "critique_prompt": critique_prompt,
                    "critique_res": critique_res,
                })
                logger.info(format_log_critique(critique_res))
                gc.collect()
                torch.cuda.empty_cache()
                                    
                """
                Revision
                """
                logger.info(f"     |----- [iteration {j}] Revision")
                revision_prompt, revision_raw_response, revised_code, revised_code_path, revised_img_path, revised_pdf_path, revised_combined_image_path = revision_agent.act(
                    ground_truth_img_path, critique_res, output_path, output_prefix, output_suffix)
                logger.info(f"     |----- [iteration {j}] Revised Evaluation")
                
                revised_code_evaluate_path = f"{os.path.splitext(revised_code_path)[0]}_evaluate.py"
                new_evaluation_results, new_evaluation_revision_rank = evaluation_agent.act(
                    ground_truth_code_path, revised_code_evaluate_path)
                
                evaluation_improved = True if new_evaluation_results.get(lowest_metric, 0) > evaluation_results.get(lowest_metric, 0) else False
                logger.info(f"            >>> evaluation improved: {evaluation_improved}")
                
                
                logger.info(f"     |----- [iteration {j}] Revised Verification")
                new_verification_results, new_revision_rank = verification_agent.act(
                    ground_truth_img_path, revised_img_path)
                
                for metric, value in new_verification_results.items():
                    old_value = verification_results.get(metric, 0)
                    changed = value - old_value
                    logger.info(f"            >>> {metric}: {value} ({'+' if changed >=0 else ''}{changed})")
                
                
                improved = True if new_verification_results.get(lowest_metric, 0) > verification_results.get(lowest_metric, 0) else False
                logger.info(f"            >>> verification improved: {improved}")
                
                log_entry = {
                    "iteration": j,
                    "agent": "RevisionAgent",
                    "revision_prompt": revision_prompt,
                    "revision_raw_response": revision_raw_response,
                    "revised_code": revised_code,
                    "revised_code_path": revised_code_path,
                    "revised_img_path": revised_img_path,
                    "revised_pdf_path": revised_pdf_path,
                    "revised_combined_image_path": revised_combined_image_path,
                    "new_evaluation_results": new_evaluation_results,
                    "new_verification_results": new_verification_results,
                    "improved": improved,
                    "evaluation_improved": evaluation_improved,
                }
                log.append(log_entry)
                gc.collect()
                torch.cuda.empty_cache()
            
                if improved:
                    verification_results = new_verification_results
                    revision_rank = new_revision_rank
                    generated_code = revised_code
                    generated_code_path = revised_code_path
                    combined_image_path = revised_combined_image_path
                    
                    old_avg = sum(evaluation_results.values()) / len(evaluation_results)
                    new_avg = sum(new_evaluation_results.values()) / len(new_evaluation_results)
                    improvement = new_avg - old_avg
                    if improvement > 0:
                        improved_flg = True
                        avg_improvement = max(improvement, avg_improvement)
                        evaluation_results = new_evaluation_results
                    
            except Exception as e:
                logger.error(f"    ** Error: {e} **", exc_info=True)
                log.append({
                    "iteration": j,
                    "agent": "Error",
                    "error": str(e),
                })
                continue
            finally:
                with open(f"{output_path}{output_suffix}_log.json", "w") as f:
                    json.dump(log, f, indent=2)
        
        return {"data": output_suffix, "log": log, "flag": "improved" if improved_flg else "not_improved", "improvement": avg_improvement}
    
    except Exception as e:
        logger.error(f"## Failed to process data {output_suffix}: {e} ##", exc_info=True)
        log.append({
            "data": output_suffix,
            "agent": "Error",
            "error": str(e),
        })
        return {"data": output_suffix, "log": log, "flag": "error", "improvement": 0}

# 3. Metal-v: GenerationAgent, VisualCritiqueAgent, VisualRevisionAgent, VerificationAgent, EvaluationAgent
def visual_only_critique_process_data(data_item, config, model, max_iterations):
   
    log = []
    
    output_suffix = os.path.splitext(os.path.basename(data_item["GroundTruthFigure"]))[0]
    output_prefix = "direct_generation"
    output_path = f"{config['working_dir']}{output_suffix}/"
    os.makedirs(output_path, exist_ok=True)
    
    logger = setup_logger(output_path, output_suffix)
    
    try:
        logger.info(f"## Processing data {output_suffix} ##")
        
        ground_truth_img_path = os.path.join(config["img_parent_dir"], data_item["GroundTruthFigure"])
        ground_truth_code_path = os.path.join(config["img_parent_dir"], data_item["GroundTruthFigureCode"])
        generation_prompt = data_item["Instruction"]
        
        agents = {
            "GenerationAgent": GenerationAgent(model, **config.get('agent_kwargs', {})),
            "VisualCritiqueAgent": VisualCritiqueAgent(model, **config.get('agent_kwargs', {})),
            "VisualRevisionAgent": VisualRevisionAgent(model, **config.get('agent_kwargs', {})),
            "EvaluationAgent": EvaluationAgent(model, **config.get('agent_kwargs', {})),
            "VerificationAgent": VerificationAgent(model, **config.get('agent_kwargs', {})),
        }
        generation_agent = agents["GenerationAgent"]
        visual_critique_agent = agents["VisualCritiqueAgent"]
        visual_revision_agent = agents["VisualRevisionAgent"]
        verification_agent = agents["VerificationAgent"]
        evaluation_agent = agents["EvaluationAgent"]
        
        
        """
        Direct Generation
        """
        logger.info("  |-- Direct Generation")
       
        raw_response, generated_code, generated_code_path, generated_image_path, combined_image_path = generation_agent.act(
            generation_prompt, ground_truth_img_path, output_path, output_prefix, output_suffix)
        gc.collect()
        torch.cuda.empty_cache()
        
        logger.info("  |-- Direct Evaluation")
        generated_code_evaluate_path = f"{os.path.splitext(generated_code_path)[0]}_evaluate.py"
        evaluation_results, evaluate_revision_rank = evaluation_agent.act(
            ground_truth_code_path, generated_code_evaluate_path)
        log_entry = {
            "iteration": 0,
            "agent": "GenerationAgent",
            "generation_prompt": generation_prompt,
            "raw_response": raw_response,
            "generated_code": generated_code,
            "generated_code_path": generated_code_path,
            "combined_image_path": combined_image_path,
            "evaluation_results": evaluation_results,
        }
        log.append(log_entry)
        for metric, value in evaluation_results.items():
            logger.info(f"      >>> {metric}: {value}")
            
        logger.info("  |-- Direct Verification")
        verification_results, revision_rank = verification_agent.act(
            ground_truth_img_path, generated_image_path)
        log.append({
            "iteration": 0,
            "agent": "VerificationAgent",
            "verification_results": verification_results,
        })
        for metric, value in verification_results.items():
            logger.info(f"      >>> {metric}: {value}")
        
        if not revision_rank or max_iterations == -1:
            with open(f"{output_path}{output_suffix}_log.json", "w") as f:
                        json.dump(log, f, indent=2)
            return {"data": output_suffix, "log": log, "flag": "perfect"}
        
        error_file_path = f"{os.path.splitext(generated_code_path)[0]}_error.txt"
        if os.path.exists(error_file_path):
            with open(error_file_path) as f:
                error = f.read()
            logger.warning(f"Detected error: {error}. Try Again.")
            os.system(f"rm {error_file_path}")
            log = []
            
            logger.info("  |-- Direct Generation (Try Again)")
       
            raw_response, generated_code, generated_code_path, generated_image_path, combined_image_path = generation_agent.act(
                generation_prompt, ground_truth_img_path, output_path, output_prefix, output_suffix)
            gc.collect()
            torch.cuda.empty_cache()
            
            logger.info("  |-- Direct Evaluation (Try Again)")
            generated_code_evaluate_path = f"{os.path.splitext(generated_code_path)[0]}_evaluate.py"
            evaluation_results, evaluate_revision_rank = evaluation_agent.act(
                ground_truth_code_path, generated_code_evaluate_path)
            log_entry = {
                "iteration": 0,
                "agent": "GenerationAgent",
                "generation_prompt": generation_prompt,
                "raw_response": raw_response,
                "generated_code": generated_code,
                "generated_code_path": generated_code_path,
                "combined_image_path": combined_image_path,
                "evaluation_results": evaluation_results,
            }
            log.append(log_entry)
            for metric, value in evaluation_results.items():
                logger.info(f"      >>> {metric}: {value}")
            
            logger.info("  |-- Direct Verification (Try Again)")
            verification_results, revision_rank = verification_agent.act(
                ground_truth_img_path, generated_image_path)
            log.append({
                "iteration": 0,
                "agent": "VerificationAgent",
                "verification_results": verification_results,
            })
            for metric, value in verification_results.items():
                logger.info(f"      >>> {metric}: {value}")
            
            if not revision_rank or max_iterations == -1:
                with open(f"{output_path}{output_suffix}_log.json", "w") as f:
                            json.dump(log, f, indent=2)
                return {"data": output_suffix, "log": log, "flag": "perfect"}
            
            error_file_path = f"{os.path.splitext(generated_code_path)[0]}_error.txt"
            if os.path.exists(error_file_path):
                with open(error_file_path) as f:
                    error = f.read()
            
                log.append({
                    "iteration": 0,
                    "agent": "Detected Error, Skip Iterative Critique and Revision",
                    "error": error,
                })
                logger.warning(f"Detected error: {error}. Skipping iterative critique and revision.")
                return {"data": output_suffix, "log": log, "flag": "error"}
        
        
        """
        Iterative Critique and Revision
        """
        improved_flg = False
        avg_improvement = 0
        for j in range(1, max_iterations + 1):
            if not revision_rank or len(revision_rank) == 0:
                with open(f"{output_path}{output_suffix}_log.json", "w") as f:
                        json.dump(log, f, indent=2)
                break
            
            logger.info(f"   |--- Iteration {j}/{max_iterations}")
            lowest_metric = revision_rank[0]
            try:
                logger.info(f"    |---- [iteration {j}] Revising Metric: {lowest_metric} in {', '.join(revision_rank)}")
                output_prefix = f"revised_round_{j}_{lowest_metric}"
                
                """
                Visual Critique
                """
                logger.info(f"     |----- [iteration {j}] Visual Critique")
                visual_prompt, visual_res = visual_critique_agent.act(
                    lowest_metric, combined_image_path)
                log.append({
                    "iteration": j,
                    "agent": "VisualCritiqueAgent",
                    "lowest_metric": lowest_metric,
                    "visual_critique_prompt": visual_prompt,
                    "visual_critique_res": visual_res,
                })
                logger.info(format_log_critique(visual_res))
                gc.collect()
                torch.cuda.empty_cache()
                
                """
                Visual Revision
                """
                logger.info(f"     |----- [iteration {j}] Revision")
                revision_prompt, revision_raw_response, revised_code, revised_code_path, revised_img_path, revised_pdf_path, revised_combined_image_path = visual_revision_agent.act(
                    visual_res, ground_truth_img_path, generated_code, output_path, output_prefix, output_suffix)
                
                logger.info(f"     |----- [iteration {j}] Revised Evaluation")
                
                revised_code_evaluate_path = f"{os.path.splitext(revised_code_path)[0]}_evaluate.py"
                new_evaluation_results, new_evaluation_revision_rank = evaluation_agent.act(
                    ground_truth_code_path, revised_code_evaluate_path)
                
                for metric, value in new_evaluation_results.items():
                    old_value = evaluation_results.get(metric, 0)
                    changed = value - old_value
                    logger.info(f"            >>> {metric}: {value} ({'+' if changed >=0 else ''}{changed})")
                    
                evaluation_improved = True if new_evaluation_results.get(lowest_metric, 0) > evaluation_results.get(lowest_metric, 0) else False
                logger.info(f"            >>> evaluation improved: {evaluation_improved}")
                
                
                logger.info(f"     |----- [iteration {j}] Revised Verification")
                new_verification_results, new_revision_rank = verification_agent.act(
                    ground_truth_img_path, revised_img_path)
                
                for metric, value in new_verification_results.items():
                    old_value = verification_results.get(metric, 0)
                    changed = value - old_value
                    logger.info(f"            >>> {metric}: {value} ({'+' if changed >=0 else ''}{changed})")
                
                
                improved = True if new_verification_results.get(lowest_metric, 0) > verification_results.get(lowest_metric, 0) else False
                logger.info(f"            >>> verification improved: {improved}")
                
                log_entry = {
                    "iteration": j,
                    "agent": "VisualRevisionAgent",
                    "revision_prompt": revision_prompt,
                    "revision_raw_response": revision_raw_response,
                    "revised_code": revised_code,
                    "revised_code_path": revised_code_path,
                    "revised_img_path": revised_img_path,
                    "revised_pdf_path": revised_pdf_path,
                    "revised_combined_image_path": revised_combined_image_path,
                    "new_evaluation_results": new_evaluation_results,
                    "new_verification_results": new_verification_results,
                    "improved": improved,
                    "evaluation_improved": evaluation_improved,
                }
                log.append(log_entry)
                gc.collect()
                torch.cuda.empty_cache()
            
                if improved:
                    verification_results = new_verification_results
                    revision_rank = new_revision_rank
                    generated_code = revised_code
                    generated_code_path = revised_code_path
                    combined_image_path = revised_combined_image_path
                    
                    old_avg = sum(evaluation_results.values()) / len(evaluation_results)
                    new_avg = sum(new_evaluation_results.values()) / len(new_evaluation_results)
                    improvement = new_avg - old_avg
                    if improvement > 0:
                        improved_flg = True
                        avg_improvement = max(improvement, avg_improvement)
                        evaluation_results = new_evaluation_results
                    
            except Exception as e:
                logger.error(f"    ** Error: {e} **", exc_info=True)
                log.append({
                    "iteration": j,
                    "agent": "Error",
                    "error": str(e),
                })
                continue
            finally:
                with open(f"{output_path}{output_suffix}_log.json", "w") as f:
                    json.dump(log, f, indent=2)
        
        return {"data": output_suffix, "log": log, "flag": "improved" if improved_flg else "not_improved", "improvement": avg_improvement}
    
    except Exception as e:
        logger.error(f"## Failed to process data {output_suffix}: {e} ##", exc_info=True)
        log.append({
            "data": output_suffix,
            "agent": "Error",
            "error": str(e),
        })
        return {"data": output_suffix, "log": log, "flag": "error", "improvement": 0}

# 4. Metal-c: GenerationAgent, TextCritiqueVisualAgent, RevisionAgent, VerificationAgent, EvaluationAgent
def code_only_critique_process_data(data_item, config, model, max_iterations):
    log = []
    
    output_suffix = os.path.splitext(os.path.basename(data_item["GroundTruthFigure"]))[0]
    output_prefix = "direct_generation"
    output_path = f"{config['working_dir']}{output_suffix}/"
    os.makedirs(output_path, exist_ok=True)
    
    logger = setup_logger(output_path, output_suffix)
    
    try:
        logger.info(f"## Processing data {output_suffix} ##")
        
        ground_truth_img_path = os.path.join(config["img_parent_dir"], data_item["GroundTruthFigure"])
        ground_truth_code_path = os.path.join(config["img_parent_dir"], data_item["GroundTruthFigureCode"])
        generation_prompt = data_item["Instruction"]
        
        agents = {
            "GenerationAgent": GenerationAgent(model, **config.get('agent_kwargs', {})),
            "TextCritiqueVisualAgent": TextCritiqueVisualAgent(model, **config.get('agent_kwargs', {})),
            "RevisionAgent": RevisionAgent(model, **config.get('agent_kwargs', {})),
            "EvaluationAgent": EvaluationAgent(model, **config.get('agent_kwargs', {})),
            "VerificationAgent": VerificationAgent(model, **config.get('agent_kwargs', {})),
        }
        generation_agent = agents["GenerationAgent"]
        text_critique_visual_agent = agents["TextCritiqueVisualAgent"]
        revision_agent = agents["RevisionAgent"]
        verification_agent = agents["VerificationAgent"]
        evaluation_agent = agents["EvaluationAgent"]
        
        
        """
        Direct Generation
        """
        logger.info("  |-- Direct Generation")
       
        raw_response, generated_code, generated_code_path, generated_image_path, combined_image_path = generation_agent.act(
            generation_prompt, ground_truth_img_path, output_path, output_prefix, output_suffix)
        gc.collect()
        torch.cuda.empty_cache()
        
        logger.info("  |-- Direct Evaluation")
        generated_code_evaluate_path = f"{os.path.splitext(generated_code_path)[0]}_evaluate.py"
        evaluation_results, evaluate_revision_rank = evaluation_agent.act(
            ground_truth_code_path, generated_code_evaluate_path)
        log_entry = {
            "iteration": 0,
            "agent": "GenerationAgent",
            "generation_prompt": generation_prompt,
            "raw_response": raw_response,
            "generated_code": generated_code,
            "generated_code_path": generated_code_path,
            "combined_image_path": combined_image_path,
            "evaluation_results": evaluation_results,
        }
        log.append(log_entry)
        for metric, value in evaluation_results.items():
            logger.info(f"      >>> {metric}: {value}")
            
        logger.info("  |-- Direct Verification")
        verification_results, revision_rank = verification_agent.act(
            ground_truth_img_path, generated_image_path)
        log.append({
            "iteration": 0,
            "agent": "VerificationAgent",
            "verification_results": verification_results,
        })
        for metric, value in verification_results.items():
            logger.info(f"      >>> {metric}: {value}")
        
        if not revision_rank or max_iterations == -1:
            with open(f"{output_path}{output_suffix}_log.json", "w") as f:
                        json.dump(log, f, indent=2)
            return {"data": output_suffix, "log": log, "flag": "perfect"}
        
        error_file_path = f"{os.path.splitext(generated_code_path)[0]}_error.txt"
        if os.path.exists(error_file_path):
            with open(error_file_path) as f:
                error = f.read()
            logger.warning(f"Detected error: {error}. Try Again.")
            os.system(f"rm {error_file_path}")
            log = []
            
            logger.info("  |-- Direct Generation (Try Again)")
       
            raw_response, generated_code, generated_code_path, generated_image_path, combined_image_path = generation_agent.act(
                generation_prompt, ground_truth_img_path, output_path, output_prefix, output_suffix)
            gc.collect()
            torch.cuda.empty_cache()
            
            logger.info("  |-- Direct Evaluation (Try Again)")
            generated_code_evaluate_path = f"{os.path.splitext(generated_code_path)[0]}_evaluate.py"
            evaluation_results, evaluate_revision_rank = evaluation_agent.act(
                ground_truth_code_path, generated_code_evaluate_path)
            log_entry = {
                "iteration": 0,
                "agent": "GenerationAgent",
                "generation_prompt": generation_prompt,
                "raw_response": raw_response,
                "generated_code": generated_code,
                "generated_code_path": generated_code_path,
                "combined_image_path": combined_image_path,
                "evaluation_results": evaluation_results,
            }
            log.append(log_entry)
            for metric, value in evaluation_results.items():
                logger.info(f"      >>> {metric}: {value}")
            
            logger.info("  |-- Direct Verification (Try Again)")
            verification_results, revision_rank = verification_agent.act(
                ground_truth_img_path, generated_image_path)
            log.append({
                "iteration": 0,
                "agent": "VerificationAgent",
                "verification_results": verification_results,
            })
            for metric, value in verification_results.items():
                logger.info(f"      >>> {metric}: {value}")
            
            if not revision_rank or max_iterations == -1:
                with open(f"{output_path}{output_suffix}_log.json", "w") as f:
                            json.dump(log, f, indent=2)
                return {"data": output_suffix, "log": log, "flag": "perfect"}
            
            error_file_path = f"{os.path.splitext(generated_code_path)[0]}_error.txt"
            if os.path.exists(error_file_path):
                with open(error_file_path) as f:
                    error = f.read()
            
                log.append({
                    "iteration": 0,
                    "agent": "Detected Error, Skip Iterative Critique and Revision",
                    "error": error,
                })
                logger.warning(f"Detected error: {error}. Skipping iterative critique and revision.")
                return {"data": output_suffix, "log": log, "flag": "error"}
        
        
        """
        Iterative Critique and Revision
        """
        improved_flg = False
        avg_improvement = 0
        for j in range(1, max_iterations + 1):
            if not revision_rank or len(revision_rank) == 0:
                with open(f"{output_path}{output_suffix}_log.json", "w") as f:
                        json.dump(log, f, indent=2)
                break
            
            logger.info(f"   |--- Iteration {j}/{max_iterations}")
            try:
                lowest_metric = revision_rank[0]
                output_prefix = f"revised_round_{j}"
                
                
                """
                Text Critique (based on visual)
                """
                logger.info(f"     |----- [iteration {j}] Code Critique")
                text_prompt, text_res = text_critique_visual_agent.act(
                        ground_truth_img_path, generated_code)
                log.append({
                    "iteration": j,
                    "agent": "TextCritiqueVisualAgent",
                    "lowest_metric": lowest_metric,
                    "text_critique_prompt": text_prompt,
                    "text_critique_res": text_res,
                })
                logger.info(format_log_critique(text_res))
                gc.collect()
                torch.cuda.empty_cache()
                                    
                """
                Revision
                """
                logger.info(f"     |----- [iteration {j}] Revision")
                revision_prompt, revision_raw_response, revised_code, revised_code_path, revised_img_path, revised_pdf_path, revised_combined_image_path = revision_agent.act(
                    ground_truth_img_path, text_res, output_path, output_prefix, output_suffix)
                
                logger.info(f"     |----- [iteration {j}] Revised Evaluation")
                
                revised_code_evaluate_path = f"{os.path.splitext(revised_code_path)[0]}_evaluate.py"
                new_evaluation_results, new_evaluation_revision_rank = evaluation_agent.act(
                    ground_truth_code_path, revised_code_evaluate_path)
                
                for metric, value in new_evaluation_results.items():
                    old_value = evaluation_results.get(metric, 0)
                    changed = value - old_value
                    logger.info(f"            >>> {metric}: {value} ({'+' if changed >=0 else ''}{changed})")
                    
                evaluation_improved = True if new_evaluation_results.get(lowest_metric, 0) > evaluation_results.get(lowest_metric, 0) else False
                logger.info(f"            >>> evaluation improved: {evaluation_improved}")
                
                
                logger.info(f"     |----- [iteration {j}] Revised Verification")
                new_verification_results, new_revision_rank = verification_agent.act(
                    ground_truth_img_path, revised_img_path)
                
                for metric, value in new_verification_results.items():
                    old_value = verification_results.get(metric, 0)
                    changed = value - old_value
                    logger.info(f"            >>> {metric}: {value} ({'+' if changed >=0 else ''}{changed})")
                
                
                improved = True if new_verification_results.get(lowest_metric, 0) > verification_results.get(lowest_metric, 0) else False
                logger.info(f"            >>> verification improved: {improved}")
                
                log_entry = {
                    "iteration": j,
                    "agent": "RevisionAgent",
                    "revision_prompt": revision_prompt,
                    "revision_raw_response": revision_raw_response,
                    "revised_code": revised_code,
                    "revised_code_path": revised_code_path,
                    "revised_img_path": revised_img_path,
                    "revised_pdf_path": revised_pdf_path,
                    "revised_combined_image_path": revised_combined_image_path,
                    "new_evaluation_results": new_evaluation_results,
                    "new_verification_results": new_verification_results,
                    "improved": improved,
                    "evaluation_improved": evaluation_improved,
                }
                log.append(log_entry)
                gc.collect()
                torch.cuda.empty_cache()
            
                if improved:
                    verification_results = new_verification_results
                    revision_rank = new_revision_rank
                    generated_code = revised_code
                    generated_code_path = revised_code_path
                    combined_image_path = revised_combined_image_path
                    
                    old_avg = sum(evaluation_results.values()) / len(evaluation_results)
                    new_avg = sum(new_evaluation_results.values()) / len(new_evaluation_results)
                    improvement = new_avg - old_avg
                    if improvement > 0:
                        improved_flg = True
                        avg_improvement = max(improvement, avg_improvement)
                        evaluation_results = new_evaluation_results
                    
            except Exception as e:
                logger.error(f"    ** Error: {e} **", exc_info=True)
                log.append({
                    "iteration": j,
                    "agent": "Error",
                    "error": str(e),
                })
                continue
            finally:
                with open(f"{output_path}{output_suffix}_log.json", "w") as f:
                    json.dump(log, f, indent=2)
        
        return {"data": output_suffix, "log": log, "flag": "improved" if improved_flg else "not_improved", "improvement": avg_improvement}
    
    except Exception as e:
        logger.error(f"## Failed to process data {output_suffix}: {e} ##", exc_info=True)
        log.append({
            "data": output_suffix,
            "agent": "Error",
            "error": str(e),
        })
        return {"data": output_suffix, "log": log, "flag": "error", "improvement": 0}


class MultiAgentSystem:
    def __init__(self, model, data, n_process, config, **kwargs):
        self.model = model
        self.data = data
        self.n_process = n_process
        self.config = config
        self.config.update(kwargs)
        self.system = config.get("agent_kwargs",{}).get("system", "Metal")
        self.max_iterations = config["max_iterations"]
        self.logs = []
        
        self.main_logger = logging.getLogger("MultiAgentSystem")
        self.main_logger.setLevel(logging.INFO)
        main_log_file = os.path.join(self.config["working_dir"], f"{self.config['exp_name']}_main.log")
        fh = logging.FileHandler(main_log_file)
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        self.main_logger.addHandler(fh)
    
    def run(self):
        num_workers = self.n_process
        self.main_logger.info(f"Prcoessing {len(self.data)} data using {num_workers} workers.")
        
        ctx = multiprocessing.get_context('spawn')
        
        with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as executor:
            if self.system == "Metal-s":
                func = partial(sinlge_critique_process_data, config=self.config, model=self.model, max_iterations=self.max_iterations)
            elif self.system == "Metal-v":
                func = partial(visual_only_critique_process_data, config=self.config, model=self.model, max_iterations=self.max_iterations)
            elif self.system == "Metal-c":
                func = partial(code_only_critique_process_data, config=self.config, model=self.model, max_iterations=self.max_iterations)
            else:
                func = partial(process_data, config=self.config, model=self.model, max_iterations=self.max_iterations)
            future_to_data = {executor.submit(func, data_item): data_item for data_item in self.data}
            
            for future in as_completed(future_to_data):
                data_item = future_to_data[future]
                try:
                    result = future.result()
                    self.logs.append(result)
                    flag = result["flag"]
                    improvment = result.get("improvement", 0)
                    if flag == "improved":
                        self.main_logger.info(f"{result['data']} Completed. Result: {flag} (+{improvment})")
                    else:
                        self.main_logger.info(f"{result['data']} Completed. Result: {flag}")
                except Exception as exc:
                    data_suffix = os.path.splitext(os.path.basename(data_item["GroundTruthFigure"]))[0]
                    self.main_logger.error(f"## Failed to process data {data_suffix}: {exc} ##", exc_info=True)
                    self.logs.append({
                        "data": data_suffix,
                        "log": [{
                            "agent": "Error",
                            "error": str(exc)
                        }],
                        "flag": "error"
                    })
        

        final_log_path = os.path.join(self.config["working_dir"], f"{self.config['exp_name']}_logs.json")
        with open(final_log_path, "w") as f:
            json.dump(self.logs, f, indent=2)
        
        self.main_logger.info(f"Processing complete. Logs saved to {final_log_path}")


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="../dataset")
    parser.add_argument("--working_dir", type=str, default="working")
    parser.add_argument("--exp_name", type=str, default="exp1")
    parser.add_argument("--max_iter", type=int, default=3)
    parser.add_argument("--n_process", type=int, default=8)
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument("--data_range", type=str, default="100-200")
    parser.add_argument("--agent_kwargs", type=json.loads, default="{}")
    args = parser.parse_args()
    
    working_dir = args.working_dir
    exp_name = args.exp_name
    max_iterations = args.max_iter
    n_process = args.n_process
    model = args.model
    data_range = args.data_range.split("-")
    start, end = int(data_range[0]), int(data_range[1])
    kwargs = args.agent_kwargs
    
    img_parent_dir = args.dataset_dir
    test_data_path = os.path.join(img_parent_dir, "test.jsonl")
    with open(test_data_path) as f:
        test_data = [json.loads(line) for line in f]
    data = test_data[start:end]
    
    os.makedirs(working_dir, exist_ok=True)
    
    # agent_kwargs = {
    # "cuda_device": "0",
    # "system: Metal, Metal-s, Metal-r"
    # }
    
    # System:
    # 1. Metal: GenerationAgent, TextCritiqueAgent, VisualCritiqueAgent, RevisionAgent, VerificationAgent, EvaluationAgent
    # 2. Metal-s: GenerationAgent, SingleCritiqueAgent, RevisionAgent, VerificationAgent, EvaluationAgent
    # 3. Metal-v: GenerationAgent, VisualCritiqueAgent, VisualRevisionAgent, VerificationAgent, EvaluationAgent
    # 4. Metal-c: GenerationAgent, TextCritiqueVisualAgent, RevisionAgent, VerificationAgent, EvaluationAgent
    
    config = {
        "exp_name": exp_name,
        "img_parent_dir": img_parent_dir,
        "working_dir": working_dir,
        "max_iterations": max_iterations,
        "agent_kwargs": kwargs,
    }
    
    mas = MultiAgentSystem(model, data, n_process, config)
    mas.run()
