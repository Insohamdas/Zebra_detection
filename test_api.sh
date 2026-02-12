#!/bin/bash
# Test script for Zebra Re-ID Flask API

API_URL="http://127.0.0.1:5001"

echo "=================================="
echo "Zebra Re-ID API Test Suite"
echo "=================================="
echo

echo "1. Health Check"
echo "---------------"
curl -s "$API_URL/health" | python -m json.tool
echo
echo

echo "2. System Statistics"
echo "--------------------"
curl -s "$API_URL/stats" | python -m json.tool
echo
echo

echo "3. Get Zebra Metadata (FAISS ID 0)"
echo "-----------------------------------"
curl -s "$API_URL/zebra/0" | python -m json.tool
echo
echo

echo "4. Identify Zebra (FAISS ID 0 - should match itself)"
echo "-----------------------------------------------------"
curl -s -X POST "$API_URL/identify" \
  -H "Content-Type: application/json" \
  -d '{"faiss_id": 0}' | python -m json.tool
echo
echo

echo "5. Identify Zebra (FAISS ID 5 - should match itself)"
echo "-----------------------------------------------------"
curl -s -X POST "$API_URL/identify" \
  -H "Content-Type: application/json" \
  -d '{"faiss_id": 5}' | python -m json.tool
echo
echo

echo "6. Identify Zebra (FAISS ID 100)"
echo "---------------------------------"
curl -s -X POST "$API_URL/identify" \
  -H "Content-Type: application/json" \
  -d '{"faiss_id": 100}' | python -m json.tool
echo
echo

echo "=================================="
echo "All tests completed!"
echo "=================================="
