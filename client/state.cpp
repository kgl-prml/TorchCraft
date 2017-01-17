/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "state.h"

#include "BWEnv/fbs/messages_generated.h"

extern "C" {
#include <TH/TH.h>
#include <lauxlib.h>
#include <luaT.h>
#include <lualib.h>
}

namespace {

template <typename OutputIt>
void copyIntegerArray(lua_State* L, int index, OutputIt it, int max = -1) {
  lua_pushvalue(L, index);
  int i = 1;
  while (max < 0 || i <= max) {
    lua_rawgeti(L, -1, i);
    if (lua_isnil(L, -1)) {
      lua_pop(L, 1);
      break;
    }
    *it++ = luaL_checkint(L, -1);
    lua_pop(L, 1);
    i++;
  }
  lua_pop(L, 1);
}

template <typename OutputIt>
void copy2DIntegerArray(lua_State* L, int index, OutputIt it, int size[2]) {
  lua_pushvalue(L, index);

  int i = 1;
  int maxj = -1;
  while (true) {
    lua_rawgeti(L, -1, i);
    if (lua_isnil(L, -1) || !lua_istable(L, -1)) {
      lua_pop(L, 1);
      break;
    }

    int j = 1;
    while (true) {
      lua_rawgeti(L, -1, j);
      if (lua_isnil(L, -1)) {
        lua_pop(L, 1);
        break;
      }

      *it++ = luaL_checkint(L, -1);
      lua_pop(L, 1);
      j++;
    }

    lua_pop(L, 1);
    i++;

    if (maxj < 0) {
      maxj = j;
    } else if (maxj != j) {
      break;
    }
  }

  size[0] = i - 1;
  size[1] = maxj - 1;
  lua_pop(L, 1);
}

void copyMapData(
    lua_State* L,
    int index,
    std::vector<uint8_t>& dest,
    int size[2]) {
  THByteTensor* data =
      static_cast<THByteTensor*>(luaT_checkudata(L, index, "torch.ByteTensor"));
  assert(THByteTensor_nDimension(data) == 2);
  auto storage = data->storage;
  auto n = THByteStorage_size(storage);
  dest.resize(n);
  memcpy(dest.data(), storage->data, n);
  size[0] = THByteTensor_size(data, 0);
  size[1] = THByteTensor_size(data, 1);
}

} // namespace

namespace client {

State::State() : RefCounted(), frame(new replayer::Frame()) {
  reset();
}

State::~State() {
  frame->decref();
}

void State::reset() {
  lag_frames = 0;
  map_data.clear();
  map_data_size[0] = 0;
  map_data_size[1] = 0;
  map_name.clear();
  frame_string.clear();
  deaths.clear();
  frame_from_bwapi = 0;
  battle_frame_count = 0;
  game_ended = false;
  game_won = false;
  img_mode.clear();
  screen_position[0] = -1;
  screen_position[1] = -1;
  image.clear(); // XXX invalidates existing tensors pointing to it
  image_size[0] = 0;
  image_size[1] = 0;
}

std::vector<std::string> State::update(
    const TorchCraft::HandshakeServer* handshake) {
  std::vector<std::string> upd;
  lag_frames = handshake->lag_frames();
  upd.emplace_back("lag_frames");
  if (flatbuffers::IsFieldPresent(
          handshake, TorchCraft::HandshakeServer::VT_MAP_DATA)) {
    map_data.assign(
        handshake->map_data()->begin(), handshake->map_data()->end());
    upd.emplace_back("map_data");
  }
  if (flatbuffers::IsFieldPresent(
          handshake, TorchCraft::HandshakeServer::VT_MAP_SIZE)) {
    map_data_size[0] = handshake->map_size()->x();
    map_data_size[1] = handshake->map_size()->y();
  }
  if (flatbuffers::IsFieldPresent(
          handshake, TorchCraft::HandshakeServer::VT_MAP_NAME)) {
    map_name = handshake->map_name()->str();
    upd.emplace_back("map_name");
  }
  // TODO: is_replay
  player_id = handshake->player_id();
  upd.emplace_back("player_id");
  neutral_id = handshake->neutral_id();
  upd.emplace_back("neutral_id");
  battle_frame_count = handshake->battle_frame_count();
  upd.emplace_back("battle_frame_count");
  return upd;
}

std::vector<std::string> State::update(const TorchCraft::Frame* frame) {
  std::vector<std::string> upd;

  if (flatbuffers::IsFieldPresent(frame, TorchCraft::Frame::VT_DATA)) {
    frame_string.assign(frame->data()->begin(), frame->data()->end());
    std::istringstream ss(frame_string);
    ss >> *this->frame;
    upd.emplace_back("frame_string");
    upd.emplace_back("frame");
  }

  deaths.clear();
  upd.emplace_back("deaths");
  if (flatbuffers::IsFieldPresent(frame, TorchCraft::Frame::VT_DEATHS)) {
    deaths.assign(frame->deaths()->begin(), frame->deaths()->end());
  }

  frame_from_bwapi = frame->frame_from_bwapi();
  upd.emplace_back("frame_from_bwapi");
  battle_frame_count = frame->battle_frame_count();
  upd.emplace_back("battle_frame_count");

  if (flatbuffers::IsFieldPresent(frame, TorchCraft::Frame::VT_IMG_MODE)) {
    img_mode = frame->img_mode()->str();
    upd.emplace_back("img_mode");
  }
  if (flatbuffers::IsFieldPresent(
          frame, TorchCraft::Frame::VT_SCREEN_POSITION)) {
    screen_position[0] = frame->screen_position()->x();
    screen_position[1] = frame->screen_position()->y();
    upd.emplace_back("screen_position");
  }
  if (flatbuffers::IsFieldPresent(frame, TorchCraft::Frame::VT_VISIBILITY) &&
      flatbuffers::IsFieldPresent(
          frame, TorchCraft::Frame::VT_VISIBILITY_SIZE)) {
    if (frame->visibility()->size() ==
        frame->visibility_size()->x() * frame->visibility_size()->y()) {
      visibility_size[0] = frame->visibility_size()->x();
      visibility_size[1] = frame->visibility_size()->y();
      visibility.assign(
          frame->visibility()->begin(), frame->visibility()->end());
      upd.emplace_back("visibility");
    } else {
      visibility_size[0] = 0;
      visibility_size[1] = 0;
      visibility.clear();
      std::cerr << "Warning: visibility data does not match visibility size"
                << std::endl;
    }
  }
  if (flatbuffers::IsFieldPresent(frame, TorchCraft::Frame::VT_IMG_DATA)) {
    updateImage(
        std::string(frame->img_data()->begin(), frame->img_data()->end()));
    upd.emplace_back("image");
  }

  return upd;
}

void State::updateImage(const std::string& msg) {
  std::istringstream ss(msg);
  std::string t;
  std::getline(ss, t, ',');
  auto width = std::stoi(t);
  std::getline(ss, t, ',');
  auto height = std::stoi(t);
  auto imgdata = msg.c_str() + ss.tellg();

  image.resize(3 * height * width);
  auto imgIt = image.begin();

  // Incoming binary data is [BGRA,...], which we transform into [R..,G..,B..].
  for (int a = 2; a >= 0; --a) {
    int it = a;
    for (int i = 0; i < height * width; i++) {
      *imgIt++ = imgdata[it];
      it += 4;
    }
  }

  image_size[0] = width;
  image_size[1] = height;
}

} // namespace client
